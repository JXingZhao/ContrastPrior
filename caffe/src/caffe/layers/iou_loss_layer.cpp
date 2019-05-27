#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/iou_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void IouLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
}

template <typename Dtype>
void IouLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->count(0, 1);
  inner_num_ = bottom[0]->count(2);
  channels_ = bottom[0]->shape(1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
}

template <typename Dtype>
void IouLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  const Dtype* prob_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = bottom[0]->count() / outer_num_;
  Dtype total_loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    Dtype loss = 0;
    vector<Dtype> inter(channels_, 0.0);
    vector<Dtype> u(channels_, 0.0);
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      for (int c = 0; c < channels_; ++c) {
        if (c == label_value) {
          inter[c] += prob_data[i * dim + c * inner_num_ + j];
          u[c] += 1;
        } else {
          u[c] += prob_data[i * dim + c * inner_num_ + j];
        }
      }
    }
    for (int c = 0; c < channels_; ++c) {
      loss += inter[c] / u[c];
    }
    total_loss += loss / channels_;
  }
  top[0]->mutable_cpu_data()[0] = total_loss / outer_num_;
}

template <typename Dtype>
void IouLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int dim = bottom[0]->count() / outer_num_;
    for (int i = 0; i < outer_num_; ++i) {
      vector<Dtype> nu_first(channels_, 0.0);
      vector<Dtype> nu_second(channels_, 0.0);
      vector<Dtype> dom(channels_, 0.0);
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        for (int c = 0; c < channels_; ++c) {
          if (c == label_value) {
            nu_first[c] += prob_data[i * dim + c * inner_num_ + j] + 1;
            nu_second[c] += prob_data[i * dim + c * inner_num_ + j];
            dom[c] += 1;
          } else {
            nu_first[c] += prob_data[i * dim + c * inner_num_ + j];
            dom[c] += prob_data[i * dim + c * inner_num_ + j];
          }
        }
      }
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < channels_; ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          for (int c = 0; c < channels_; ++c) {
            if (c == label_value) {
              bottom_diff[i * dim + c * inner_num_ + j] = (nu_first[c] - nu_second[c]) / (dom[c] * dom[c] * channels_);
            } else {
              bottom_diff[i * dim + c * inner_num_ + j] = (-nu_second[c]) / (dom[c] * dom[c] * channels_);
            }
          }
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / outer_num_;
    caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(IouLossLayer);
#endif

INSTANTIATE_CLASS(IouLossLayer);
REGISTER_LAYER_CLASS(IouLoss);

}  // namespace caffe
