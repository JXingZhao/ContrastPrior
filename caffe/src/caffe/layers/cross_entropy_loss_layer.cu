#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::list;
using std::deque;

namespace caffe {
template <typename Dtype>
__global__ void CrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const int img_label) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const Dtype gt_value = label[n * spatial_dim + s] / 255.;
    const Dtype fg_value = prob_data[n * dim + img_label * spatial_dim + s];
    const Dtype bg_value = prob_data[n * dim + 0 * spatial_dim + s];
    loss[index] = - gt_value * log(max(fg_value, Dtype(FLT_MIN))) - 
          (1 - gt_value) * log(max(bg_value, Dtype(FLT_MIN)));
  }
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int img_label = static_cast<int>(bottom[2]->cpu_data()[0]);
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  CrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, img_label);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, -1);
}

template <typename Dtype>
__global__ void CrossEntropyLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const int img_label) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const Dtype gt_value = label[n * spatial_dim + s] / 255.;
    const Dtype fg_value = bottom_diff[n * dim + img_label * spatial_dim + s];
    const Dtype bg_value = bottom_diff[n * dim + 0 * spatial_dim + s];
    bottom_diff[n * dim + img_label * spatial_dim + s] = fg_value - gt_value;
    bottom_diff[n * dim + 0 * spatial_dim + s] = bg_value - (1 - gt_value);
  }
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int img_label = static_cast<int>(bottom[2]->cpu_data()[0]);
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    CrossEntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, img_label);

    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    Dtype loss_weight = 0;
    loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, -1);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CrossEntropyLossLayer);

}  // namespace caffe
