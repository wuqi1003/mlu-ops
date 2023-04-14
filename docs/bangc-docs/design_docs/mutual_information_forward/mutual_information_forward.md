# mluopMutualInformationForward算子开发设计方案

- #### 文档基本信息

| 算子名称    | mluopMutualInformationForward |
| ---------- | -------------- |
| 编制人/日期 | 吴奇/2023-4-14 |
| 审批人/日期 | XX/2022-8-26   |
| 审批人/日期 | XX/2022-8-26   |
| 审批人/日期 | XX/2022-8-26   |

- #### 修改记录

| 版本号 | 修订人 | 修订日期   | 修订描述 |
| ------| ------| --------- | --------|
| V1.0  | 吴奇 | 2023-4-14 | 首次提交 |

- #### 内容描述

本文档为`mluopMutualInformationForward`算子的设计文档，包括需求分析、接口设计、方案设计、性能优化记录和方案实施部分。

- #### 算子需求 checklist

* 算子接口描述
* 功能描述
* 框架版本 + 对应源码路径
* 需求对应网络
* 网络中用到的规模
* 是否需要支持原位
* 是否需要支持 stride 机制
* 框架单元测试阈值指标（可选）

## 1 需求分析

### 1.1 算子需求分析

该需求分析为框架原生算子实现功能的需求分析，对于框架原生支持但 MLU-OPS 当前版本不支持的功能，需要在`1.4算子限制` 章节中显式注明。未明确注明不支持的功能，默认 MLU-OPS 全部支持。


| 算子功能简介                  | 简要填写算子功能，详细描述在 1.2 中进行说明 |
| ------------------------------| ------------------------------------------- |
| 需求来源                      | k2                    |
| 应用网络                      | 语音网络 k2 RNNT Loss（https://github.com/k2-fsa/k2） |
| 输入数据类型                  | px: float<br>py : float<br>boundary : int<br> |
| 输入 Shape                    | px : [B, S, T] or [B, S, T+1]<br/>py : [B, S+1, T]<br/>boundary : [B, 4] |
| 输入 Layout                   | px : ARRAY<br/>py: ARRAY<br>boundary : ARRAY |
| 输出数据类型                  | float                              |
| 输出 Shape                    | p : [B,S+1,T+1]<br>ans : [B]|
| 输出 Layout                   | p : ARRAY<br>ans : ARRAY|
| 模式(非输入, 算子内判断）           | modified, !modified|
| 是否含有 dim/axis 等类似语义的参数且该参数支持负数/其他特殊处理 | 无 |
| 是否含有 labels/index 等类似语义的参数且该参数支持负数/界外情况/其他特殊处理 | boundary: 形为[B, 4]，<br>boundary[b] = [begin_symbol, begin_frame, end_symbol, end_frame]，<br>表示样本b的logits和symbols序列的起止位置。 |
| 是否需要支持原位                | 否                                     |
| 是否需要支持 stride 机制        | 否                                     |
| 是否需要支持广播                | 否    |
| 0 元素检查是否直接返回          | 是                                     |
| 其他特殊需求(在线量化，融合，转数提前等，可选)| 无 |
| 本次开发优先支持的规模/模式     | 优先支持 !modified模式, 优先支持 FP32 类型 |

### 1.2 mutual_information_forward 算子功能和应用场景描述

在序列到序列任务中，如语音识别，为了解决输入和标签不对齐的问题，提出RNN-Transducer(RNNT)网络，解决上述两个问题。

RNN-Transducer是Alex Graves在2012年发表的一种基于RNN的序列到序列方案，可以将任意长度的输入序列转换到任意长度的输出序列。

mutual_information_forward 即为k2 rnnt_loss 中计算互信息 ans 的自定义kernel。

#### 1.2.1 k2 rnnt_loss 正向计算过程

(1) 计算logsumexp、px、py

根据规模为(B,T,S+1,C)的输入 logits tensor，计算分母logsumexp。所以有：

```math
\begin{aligned}

denom = logits(t,s,k)^{max} + \log{\sum_{k \in C}e^{logits(t,s,k) - logits(t,s,k)^{max}}}

\end{aligned}
```

**计算logsumexp、px、py 过程在 k2 rnnt_loss 内通过调用 pytorch 自带 api完成实现。**

(2) 计算 $p(s,t)$ ,

如果需要递归得到上面$p(s,t)$，需要在 $p(s=s_{begin},t=t_{begin})=0$的基础上，先行计算出两组边缘数据：

(2.1) 每个t时刻的输出为起始字符的概率信息 p(s=s_begin, t)；

(2.2) t起始时刻每个目标字符输出的概率信息 p(s, t=t_begin);

然后可按照递归乘加方法获得$p(s,t)$:

$p(s,t) = \ p(s,t-1) \cdot \emptyset(s,t-1) + \ p(s-1,t) \cdot y(s-1, t)$

(3) 计算 ans

根据前面公式推导，可知 ans 可由正向$p(s_{end},t_{end})$得到，

* $p(y^|x)=\sum^{s:s_{begin}-s_{end}}_{t:t_{begin}-t_{end}} \ p(t,u) = \ p(s_{end},t_{end}) $ 

参考竞品，由 $p$ 计算得到的 $ln(p(y|x))$ 命名 ans，

```math
\begin{aligned} 
ans = p(s_end,t_end)
\end{aligned}
```
以上(2)(3)两部分即为 mutual_information_forward 自定义kernel主要功能。

#### 1.2.2 mutual_information_forward 分析
- Input：
  - px: 传输symbol的概率。形为 [B,S,T]\(modified模式\) or [B,S,T+1]\(非modified模式\)。
  - py: 传输结束符(termination_symbol)的概率，形为 [B,S+1,T]。
  - boundary:各batch下symbol和frame起始与结束index，形为[B,4]，boundary[b][4]=[begin_symbol, begin_frame, end_symbol, end_frame]。

- output：
  - p: 各frame对应每个symbol的概率信息，形为[B,S+1,T+1]
  - ans: 每个batch内最终px-py互信息，由 p 获得，形为[B]。

- 计算过程：
  - 1. 由输入参数boundary获得各batch下的$s_{begin}, s_{end}, t_{begin}, t_{end}$ 范围；
  - 2. 由输入参数 px、py 及上面的 $s_{begin}, s_{end}, t_{begin}, t_{end}$ 计算得到各个$p(s,t)$;
  - 3. 由最终的 $p(s_{end}, t_{end})$ 获得最终互信息 $ans$ 。

### 1.3 算子输入输出参数要求

|参数|语义|类型(输入/输出)|支持类型|物理布局|规模说明|
|---|----|-------------|-------|------|-------|
|handle|句柄|输入|cnnlHandle_t | / |无|
|mutual_information_forward_desc|存放算子信息结构体| 输入 |cnnlMutualInformationForwardDescriptor_t |	 / | 无 |
|px_desc | 对输入信息的描述，包含维度、布局和数据类型信息 | 输入  |         	cnnlTensorDescriptor_t   |	/     |  	无    |
| px  |     	输入概率数据    |  输入  | float |  ARRAY  | 	3D Tensor	|
|py_desc | 对输入信息的描述，包含维度、布局和数据类型信息 | 输入 |          	cnnlTensorDescriptor_t   |	/ |      	无       	|
|py| 输入概率数据|  输入 |  float | ARRAY   |	3D Tensor	|
|boundary_desc |	对序列与标签起始信息的描述，包含维度、布局和数据类型信息     |      	输入  |         	cnnlTensorDescriptor_t   |	/  |     	无      |
| boundary |  序列与标签起始信息 |   输入 | int32  | ARRAY   |	2D Tensor |	
|ans_desc | 对输出互信息的描述，包含维度、布局和数据类型信息 | 输入 | cnnlTensorDescriptor_t|	/     |  	无    |   	
|ans| 输出互信息 |  输出 | 	float|  ARRAY   |	1D Tensor |	


`注意：`
1. px, py 的shape必须为3维
2. px在modified模式下为[B, S, T], 非modified模式下为[B, S, T+1]
3. py不论何种模式都是[B, S+1, T]
4. boundary 为 optional，可能为NULL，为空时default为[0, 0, S, T]


### 1.4 算子限制

|限制类型     |详细说明                                                     	|
|------------|------------------------------------------------------------|
|数据类型限制	|px、py只支持float32类型与竞品对齐 |                                      	
|布局限制    | px、py输入必须为3D Tensor，但px最低维度size可变，与modified模式相关|
|	规模限制    | 由于370和590上nram空间大小不同，因此规模大小限制也不同<br>在370上，(T+1) \* (S+1) ≤ 54612<br>在590上，(T+1) \* (S+1) ≤ 32768；	|                               	
|功能限制     |	boundary中只包含为正整数，若出现负数则算子的功能是未定义的行为|	
|功能限制     |	当输入含有NaN/INF时，此算子的功能是未定义的行为|
|功能限制     |	不支持原位|                                                   	
|功能限制     |	不支持stride机制|                                             	
|功能限制     |	不支持广播|
                                                   
`备注：`

* 当前算子功能和k2 release v1.23.4 功能对齐

### 1.5 验收标准

#### 1.5.1 精度验收标准（不同算子验收标准见外网wiki=52441150）
* CNNL精度验收标准：该算子为复合类算子，采用当前的diff1 diff2评价公式，验收标准为3e-3。由于该算子不支持double，因此不使用静态阈值
 

#### 1.5.2 性能验收标准
* 本次优先交付功能，后续视情况再优化性能；

## 2 算子接口设计

### 2.1 参考接口

* k2 rnnt_loss mutual_information_forward

```python
// python接口
    def forward(
        ctx,
        px: torch.Tensor,
        py: torch.Tensor,
        pxy_grads: List[Optional[torch.Tensor]],
        boundary: Optional[torch.Tensor] = None,
        return_grad: bool = False,
    );
```

```C++
// C++接口
template <typename scalar_t, int BLOCK_SIZE>
__global__ void mutual_information_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3> px,
    torch::PackedTensorAccessor32<scalar_t, 3> py,
    torch::PackedTensorAccessor32<scalar_t, 3> p,
    torch::PackedTensorAccessor32<int64_t, 2> boundary,
    torch::PackedTensorAccessor32<scalar_t, 1> ans,
    int iter);
```

### 2.2 接口设计

1. MutualInformationForward 描述符的相关接口

    (1) MutualInformationForward 描述符的创建接口

    ```c++   
    cnnlStatus_t CNNL_WIN_API cnnlCreateMutualInformationForwardDescriptor(   
      cnnlMutualInformationForwardDescriptor_t *mutual_information_forward_desc)   
    ```

    (2) MutualInformationForward 描述符的设置接口

    ```c++
    cnnlStatus_t CNNL_WIN_API cnnlSetMutualInformationForwardDescriptor(   
      cnnlMutualInformationForwardDescriptor_t mutual_information_forward_desc)
    ```

    (3) MutualInformationForward 描述符的获取接口

    ```c++   
    cnnlStatus_t CNNL_WIN_API cnnlGetMutualInformationForwardDescriptor(   
      cnnlMutualInformationForwardDescriptor_t mutual_information_forward_desc)   
    ```

    (4) MutualInformationForward 描述符的销毁接口

    ```c++   
    cnnlStatus_t CNNL_WIN_API cnnlDestroyMutualInformationForwardDescriptor(   
        cnnlMutualInformationForwardDescriptor_t mutual_information_forward_desc)   
    ```

2. MutualInformationForward 算子的接口

    ```c++  
    cnnlStatus_t CNNL_WIN_API cnnlMutualInformationForward(
      cnnlHandle_t handle,
      const cnnlMutualInformationForwardDescriptor_t mutual_information_forward_desc,
      const cnnlTensorDescriptor_t px_desc,
      const void * px,
      const cnnlTensorDescriptor_t py_desc,
      const void * py,
      const cnnlTensorDescriptor_t boundary_desc,
      const void * boundary,
      const cnnlTensorDescriptor_t p_desc,
      void * p,
      const cnnlTensorDescriptor_t ans_desc, 
      void * ans)；
    ```

## 3 实现方案设计

### 3.1 实现方案

#### 3.1 实现方案

算子的输入数据：其中B为batch，T+1 为batch中输入序列最大长度，S+1 为batch中输出序列最大长度
* px: [B, S, T] 或 [B, S, T + 1]，输出层x概率值
* py: [B, T, S+1, C]，输出层y概率值
* boundary: [B, 4]，各batch内symbol 、 frame起始-结束索引。

现有对标 torchaudio rnnt_loss 的 cnnlRNNTLoss 代码在计算alpha和loss的功能模块中，包含2个kernel，2kernel所在文件分别是

- mluBlockComputeAlphaAndLoss 
- mluBlockComputeAlphaAndLossByDiagonal3Pipeline

前者为default标量kernel文件，后者为“对角线并行动态规划 + 3级流水排布”优化kernel。


#### 3.2 伪代码实现

##### 3.2.1 标量计算 kernel —— 核间拆分batch，核内按标量计算
- 默认kernel：从 (s_begin, t_begin) 到 (s_end, t_end) 顺序标量计算：
```C++
template <typename DTYPE>
_mlu_global_ void mluBlockMutualInformationForward( const DTYPE *px,
                                                    const DTYPE *py,
                                                    const int *boundary,
                                                    const int batches,
                                                    const int max_s,
                                                    const int max_t,
                                                    const bool modified,
                                                    DTYPE *p,
                                                    DTYPE *ans){
    // 按 batch 拆分
    const int num_b_per_core = batches / taskDim;
    const int num_b_rem = batches % taskDim;
    const int num_b_cur_core = num_b_per_core + (taskId < num_b_rem);
    const int b_start = taskId * num_b_cur_core + (taskId >= num_b_rem) * num_b_rem;
    const int one_batch_num = max_s * max_t;
    int b_end = b_start + num_b_cur_core;
     
    // nram 空间划分
    DTYPE *nram_px = (DTYPE *)nram_buffer;
    DTYPE *nram_py = nram_px + (max_s - 1) * max_t;
    DTYPE *nram_p = nram_py + max_s * (max_t - 1);
 
    // 根据symbol，frame起始位置计算 s_len 与 t_len
    int s_begin = 0, s_end = max_s - 1, s_len = max_s;
    int t_begin = 0, t_end = max_t - 1, t_len = max_t;
 
    // 根据 modified 给定 px 最低维大小
    int max_sx = max_s - 1;
    int max_tx = max_t;
    int max_sy = max_s;
    int max_ty = max_t - 1;
    if(modified) {
        max_tx = max_t - 1;
    }
 
    for (int b = b_start; b < b_end; ++b) {
        // memcpy px, py form input gdram
        __memcpy_async(nram_px, px + b * max_sx * max_tx, sizeof(DTYPE), GDRAM2NRAM);
        __memcpy_async(nram_py, py + b * max_sy * max_ty, sizeof(DTYPE), GDRAM2NRAM);
 
        // update begin, end, len
        if(boundary != NONE) {
            s_begin = boundary[b][0];
            t_begin = boundary[b][1];
            s_end = boundary[b][2];
            t_end = boundary[b][3];
            s_len = s_end - s_begin + 1;
            t_len = t_end - t_begin + 1;
        }
        _asm_ volatile("sync;");
 
        // 按标量计算方式得到 p, ans
        mutualInformationForward(nram_px, nram_py, nram_p, b, s_begin, s_end, t_begin, t_end,
                                 max_sx, max_tx, max_sy, max_ty, max_s, max_y, modified, ans);
    }
}

template <typename DTYPE>
_mlu_func_ void mutualInformationForward(DTYPE *nram_px,
                                         DTYPE *nram_py,
                                         DTYPE *nram_p,
                                         const int b,
                                         const int s_begin,
                                         const int s_end,
                                         const int t_begin,
                                         const int t_end,
                                         const int max_sx,
                                         const int max_tx,
                                         const int max_sy,
                                         const int max_ty,
                                         const int max_s,
                                         const int max_t,
                                         const bool modified,
                                         DTYPE *ans) {
    // nram_p[s_begin][t_begin] = 0;
    nram_p[s_begin * max_t + t_begin] = (DTYPE)0;
     
    // compute p when t == 0
    int index_s_t = 0
    if (modified) {
        for (int s = s_begin + 1; s <= s_end; ++s) {
            nram_p[s * max_t + t_begin] = -inf;
        }
    } else {
        for (int s = s_begin + 1; s <= s_end; ++s) {
            nram_p[s * max_t + t_begin] = nram_p[(s - 1) * max_t + t_begin] + nram_px[(s - 1) * max_tx + t_begin];
        }
    }
 
    // compute p when s == 0
    for (int t = t_begin + 1; t <= t_end; ++t) {    
        nram_p[t] = nram_p[t - 1] + nram_py[t - 1];
    }
 
    // compute p when s > 0 and t > 0
    int t_off = (modified ? -1 : 0);
    for (int s = s_begin + 1; s <= s_end; ++s) {
        DTYPE p_s_t1 = nram_p[s * max_t + t_begin];
        for (int t = t_begin + 1; t <= t_end; ++t){ 
            // nram_p[s][t] = LogAdd(nram_p[s - 1][t + t_offset] + nram_px[s - 1][t + t_offset], p_s_t1 + nram_py[s][t - 1]);
            nram_p[s][t] = LogAdd(nram_p[(s - 1) * max_t + t + t_off] + nram_px[(s - 1) * max_tx + t + t_off], p_s_t1 + nram_py[s * max_ty + t - 1]);
        }
    }
 
    // get final ans
    ans[b] = nram_p[s_end * max_t + t_end];
 
    return;
}
```
<br><br>
- 对角线优化+3级流水 kernel：


```c++
template <typename DTYPE> _mlu_global_
void mluBlockMutualInformationForwardByDiagonal3Pipeline(const DTYPE *px,
                                                         const DTYPE *py,
                                                         const int *boundary,
                                                         const int batches,
                                                         const int max_s,
                                                         const int max_t,
                                                         const bool modified,
                                                         DTYPE *ans) {
    // 按 batches 拆分
    const int num_b_per_core = batches / taskDim;
    const int num_b_rem = batches % taskDim;
    const int num_b_cur_core = num_b_per_core + (taskIdY < num_b_rem);
    const int b_start = taskIdY * num_b_cur_core + (taskIdY >= num_b_rem) * num_b_rem;
    int b_end = b_start + num_b_cur_core;
 
    // nram 空间划分
    float *nram_p = (float *)nram_buffer;
    float *nram_py = nram_p + max_t * max_u;
    float *nram_px = nram_py + max_s * (max_t - 1);
    int *nram_logit_length = (int *)(nram_px + (max_s - 1) * max_t);
    int *nram_target_length = nram_logit_length + 1;
 
    // 根据symbol，frame起始位置计算 s_len 与 t_len
    int s_begin = 0, s_end = max_s - 1, s_len = max_s;
    int t_begin = 0, t_end = max_t - 1, t_len = max_t;
 
    // 根据 modified 给定 px 最低维大小
    int max_sx = max_s - 1;
    int max_tx = max_t;
    int max_sy = max_s;
    int max_ty = max_t - 1;
    if(modified) {
        max_tx = max_t - 1;
    }
 
    for (int b = b_start; b < b_end; ++b) {
        // memcpy px, py form input gdram
        __memcpy_async(nram_px, px + b * max_sx * max_tx, sizeof(DTYPE), GDRAM2NRAM);
        __memcpy_async(nram_py, py + b * max_sy * max_ty, sizeof(DTYPE), GDRAM2NRAM);
        _asm_ volatile("sync;");
 
        // update begin, end, len
        if(boundary != NONE) {
            s_begin = boundary[b][0];
            t_begin = boundary[b][1];
            s_end = boundary[b][2];
            t_end = boundary[b][3];
            s_len = s_end - s_begin + 1;
            t_len = t_end - t_begin + 1;
        }
         
        // 按对角线3级流水方式获得p,ans
        mutualInformationForwardByDiagonal3Pipeline(nram_px, nram_py, nram_p, b, s_begin, s_end, s_len, t_begin, t_end, t_len,
                                                    max_sx, max_tx, max_sy, max_ty, max_s, max_y, modified, ans);
    }
    return;
}

template <typename DTYPE> _mlu_func_
void mutualInformationForwardByDiagonal3Pipeline(DTYPE *nram_px,
                                                 DTYPE *nram_py,
                                                 DTYPE *nram_p,
                                                 const int b,
                                                 const int s_begin,
                                                 const int s_end,
                                                 const int s_len,
                                                 const int t_begin,
                                                 const int t_end,
                                                 const int t_len,
                                                 const int max_sx,
                                                 const int max_tx,
                                                 const int max_sy,
                                                 const int max_ty,
                                                 const int max_s,
                                                 const int max_t,
                                                 const bool modified,
                                                 DTYPE *ans) {
    int min_of_T_S = std::min(t_len, s_len);
    int max_of_T_S = std::max(t_len, s_len);
 
    // 2 is nram_target_lengths and -inf size
    float *max_value = nram_px + max_sx * max_tx;
    float *ping = max_value + min_of_T_S;
    float *ping_p = ping + min_of_T_S * 2 + 1; // 2 is px and py
    // 3 is px(min_of_T_S+1), py(min_of_T_S), p(min_of_T_S)
    int ping_pong_gap = min_of_T_U * 3 + 1;
 
    __bang_write_value(ping, ping_pong_gap * 2, -INFINITY);
 
    // compute first one
    nram_p[0] = (float)0;
    ping_p[ping_pong_gap] = (float)0;
 
    int repeat = t_len + s_len - 2;
    for (int i = 0; i < repeat + 2; ++i) {
        if (i < repeat) {
            pipelineLoad(nram_py, nram_px, max_sx, max_tx, max_sy, max_ty, s_begin, s_end, t_begin, t_end,
                         max_of_T_S, min_of_T_S, i, ping + (i % 2) * ping_pong_gap);
        }
        if (i > 0 && i <= repeat) {
            pipelineCompute(t_len, s_len, max_s, max_t, max_of_T_S, min_of_T_S, i, s_begin, s_end, t_begin, t_end,
                            max_value, ping_p + (i % 2) * ping_pong_gap, ping + ((i - 1) % 2) * ping_pong_gap);
        }
        if (i > 1) {
            pipelineStore(nram_p, t_len, s_len, max_s, max_t, max_of_T_S, min_of_T_S, i, ping_p + ((i - 2) % 2) * ping_pong_gap);
        }
        _asm_ volatile("sync;");
    }
 
    // get final ans
    ans[b] = nram_p[s_end * max_t + t_end];
 
    return;
}
 
 
_mlu_func_ void pipelineCompute(int T, int S, int max_s, int max_t, int max_of_T_S, int s_begin, s_end, t_begin, t_end,
                                int min_of_T_S, int i, float max_value, float *pong_p, float *ping) {
    int data_num = i < max_of_T_S ? std::min(i + 1, min_of_T_U) : T + U - i - 1;
 
    float *ping_px = ping;
    float *ping_py = ping_px + min_of_T_S + 1;
    float *ping_p = ping_py + min_of_T_S;
 
    int py_num = i < S ? data_num - 1 : data_num;
    if (py_num > 0) {
        __bang_add(ping_py, ping_py, pong_p, py_num);
    }
    int px_num = i < T ? data_num - 1 : data_num;
    float *nram_compute_p_for_px = pong_p;
    if (i >= S)  {
        nram_compute_p_for_px += 1;
    }
    if (px_num > 0){
        __bang_add(ping_emit, ping_emit, nram_compute_alpha_for_emit, emit_num);
    }
    float *nram_compute_py = ping_py;
    if (i < S) {
        nram_compute_py -= 1;
    }
 
    logSumExpVector(ping_p, nram_compute_py, ping_px, max_value, data_num);
 
}
```

### 3.3 拆分（任务拆分，多核拆分）

default 标量计算 kernel 与 对角线+3级流水 kernel都是在 B(batches) 维度进行拆分，二者不同点在于单batch内计算 p 方式的不同。

### 3.4 性能优化设计

- 在default标量计算kernel基础上，对标竞品实现了“对角线动态规划”优化kernel，性能已优于竞品；
- 在“对角线动态规划”优化kernel基础上，进一步实现了“对角线动态规划+3级流水”优化kernel，进一步优化性能；

### 3.5 可维护性设计

1、算子在B维拆分，各个core内处理自己对应batch数据，core间无耦合，方便维护；

2、关键函数命名变量命名都有充分的注释；

3、bangC 代码已加入必要log信息方便维护；

3、避免魔鬼数字。

### 3.6 测试用例设计

- 算子在网络中用到的规模：

  未提供


- 边界 case：

| px          | py        | boundary       |备注                             |
|-------------|-----------|----------------|---------------------------------|
| 4,31,1024   | 4,32,1023 | [0,0,31,1023]  |590 进入 对角线流水 kernel         |  
| 4,32,1024   | 4,33,1023 | [0,0,32,1023]  |590 进入 default 标量 kernel      |  
| 4,3,27306   | 4,4,27305 | [0,0,3,27305]  |370 进入 对角线流水 kernel         |  
| 4,3,27307   | 4,4,27306 | [0,0,3,27306]  |370 进入 default 标量 kernel      |


其他可根据需要进行补充。算子开发完毕后，补充测试报告链接。

### 3.7 算子防呆检查

* handle指针非空检查
* tensor描述符非空检查
* 描述符内成员检查：dim/datatype/layout
* 0元素检查防呆，VLOG(5)打印信息
* 内存指针检查
* 其他：根据1.4节中算子限制添加防呆


## 4 算子性能/精度问题 & 优化记录

### 4.1 当前存在问题的规模说明

首次提交，暂无。

| 提交日期  | 问题规模 | 问题描述 | 是否已修复 |
| --------- | -------- | -------- | ---------- |
| / | / | / |    /     |

### 4.2 已经过优化的规模说明

首次提交暂无

| 提交日期  | 修复规模 | 修复问题 |
| --------- | -------- | -------- |
| / | / | / |

## 5 方案实施

### 5.1 开发测试计划

- 2023.4.10 ~ 2023.4.14调研源码+设计方案 5d
- 2022.4.17 设计方案review与修改 1d
- 2022.4.17 GTest 代码开发 1d
- 2022.4.18 generatorV2 开发 1d
- 2022.4.19 host 代码开发 1d
- 2023.4.20 ~ 2023.4.21 device代码开发 2d
- 2023.4.24 ~ 2023.4.26 批量测试+测试报告 3d
- 2023.4.27 ~ 2023.4.28 提交PR+算子入库 2d


### 5.2 风险分析
NAN-INF 尚未详细测试。
