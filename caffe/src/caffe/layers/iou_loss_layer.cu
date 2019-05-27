#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/iou_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void IouLossForwardGPU(const int nthreads, const Dtype* prob_data, const Dtype* label,
          const int dim, const int spatial_dim, const bool has_ignore_label, 
          const int ignore_label, Dtype* inter_ptr, Dtype* union_ptr) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int c = (index % dim) / spatial_dim;
    const int s = (index % dim) % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label && label_value == ignore_label) {
      inter_ptr[index] = 0;
      union_ptr[index] = 0;
    } else {
      if (c == label_value) {
        inter_ptr[index] = prob_data[index];
        union_ptr[index] = 1;
      } else {
        inter_ptr[index] = 0;
        union_ptr[index] = prob_data[index];
      }
    }
  }
}

template <typename Dtype>
void IouLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* prob_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int nthreads = bottom[0]->count();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.

  Blob<Dtype> iou;
  iou.ReshapeLike(*bottom[0]);
  // avoid allocating too much memory
  Dtype* inter_ptr = iou.mutable_gpu_data();
  Dtype* union_ptr = iou.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)

  IouLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, 
      dim, inner_num_, has_ignore_label_, ignore_label_, inter_ptr, union_ptr);

  Dtype total_loss = 0;
  const Dtype* inter_cpu_data = iou.cpu_data();
  const Dtype* union_cpu_data = iou.cpu_diff();
  for (int i = 0; i < outer_num_; ++i) {
    Dtype loss = 0;
    vector<Dtype> inter_vec(channels_, 0);
    vector<Dtype> union_vec(channels_, 0);
    for (int c = 0; c < channels_; ++c) {
      for (int j = 0; j < inner_num_; ++j) {
        inter_vec[c] += inter_cpu_data[j];
        union_vec[c] += union_cpu_data[j];
      }
      inter_cpu_data += inner_num_;
      union_cpu_data += inner_num_;
    }
    for (int c = 0; c < channels_; ++c) {
      loss += inter_vec[c] / union_vec[c]; 
    }
    total_loss += loss / channels_;
  }
  top[0]->mutable_cpu_data()[0] = total_loss / outer_num_;
}

template <typename Dtype>
__global__ void IouLossStatGPU(const int nthreads, const Dtype* prob_data, const Dtype* label, const int dim,
        const int spatial_dim, const bool has_ignore_label, const int ignore_label, Dtype* num_first,
        Dtype* num_second, Dtype* denom_ptr) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int c = (index % dim) / spatial_dim;
    const int s = (index % dim) % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label && label_value == ignore_label) {
      num_first[index] = 0;
      num_second[index] = 0;
      denom_ptr[index] = 0;
    } else {
      if (c == label_value) {
        num_first[index] = prob_data[index] + 1;
        num_second[index] = prob_data[index];
        denom_ptr[index] = 1;
      } else {
        num_first[index] = prob_data[index];
        num_second[index] = 0;
        denom_ptr[index] = prob_data[index];
      }
    }
  }
}

template <typename Dtype>
__global__ void IouLossBackwardGPU(const int nthreads, Dtype* bottom_diff, const Dtype* label, const int dim,
        const int spatial_dim, const int channels, const bool has_ignore_label, const int ignore_label, const Dtype* num_first,
        const Dtype* num_second, const Dtype* denom_ptr) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int c = (index % dim) / spatial_dim;
    const int s = (index % dim) % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label && label_value == ignore_label) {
      bottom_diff[index] = 0;
    } else {
      if (c == label_value) {
        bottom_diff[index] = (num_first[n * channels + c] - num_second[n * channels + c]) / 
                (denom_ptr[n * channels + c] * denom_ptr[n * channels + c] * channels);
      } else {
        bottom_diff[index] = (-num_second[n * channels + c]) / (denom_ptr[n * channels + c] * 
                denom_ptr[n * channels + c] * channels);
      }
    }
  }
}

template <typename Dtype>
void IouLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = bottom[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = bottom[0]->count() / outer_num_;
    const int nthreads = bottom[0]->count();

    Blob<Dtype> numerator;
    Blob<Dtype> denominator;
    numerator.ReshapeLike(*bottom[0]);
    denominator.ReshapeLike(*bottom[0]);
    // avoid allocating too much memory
    Dtype* num_first = numerator.mutable_gpu_data();
    Dtype* num_second = numerator.mutable_gpu_diff();
    Dtype* denom_ptr = denominator.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    IouLossStatGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, 
            label, dim, inner_num_, has_ignore_label_, ignore_label_, num_first, num_second, denom_ptr);

    vector<int> vec_shape;
    vec_shape.push_back(outer_num_);
    vec_shape.push_back(channels_);
    vec_shape.push_back(1);
    vec_shape.push_back(1);
    Blob<Dtype> num_stat(vec_shape);
    Blob<Dtype> den_stat(vec_shape);
    Dtype* num_stat_first = num_stat.mutable_cpu_data();
    Dtype* num_stat_second = num_stat.mutable_cpu_diff();
    Dtype* den_stat_ptr = den_stat.mutable_cpu_data();
    const Dtype* num_cpu_first = numerator.cpu_data();
    const Dtype* num_cpu_second = numerator.cpu_diff();
    const Dtype* den_cpu_ptr = denominator.cpu_data();
    for (int i = 0; i < outer_num_; ++i) {
      for (int c = 0; c < channels_; ++c) {
        for (int j = 0; j < inner_num_; ++j) {
          num_stat_first[c] += num_cpu_first[j];
          num_stat_second[c] += num_cpu_second[j];
          den_stat_ptr[c] += den_cpu_ptr[j];
        }
        num_cpu_first += inner_num_;
        num_cpu_second += inner_num_;
        den_cpu_ptr += inner_num_;
      }
      num_stat_first += channels_;
      num_stat_second += channels_;
      den_stat_ptr += channels_;
    }

    const Dtype* num_gpu_first = num_stat.gpu_data();
    const Dtype* num_gpu_second = num_stat.gpu_diff();
    const Dtype* den_gpu_ptr = den_stat.gpu_data();
    IouLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_diff, 
            label, dim, inner_num_, channels_, has_ignore_label_, ignore_label_, num_gpu_first, num_gpu_second, den_gpu_ptr);
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / outer_num_;
    caffe_gpu_scal(bottom[0]->count(), loss_weight , bottom_diff);
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(IouLossLayer);

}  // namespace caffe
