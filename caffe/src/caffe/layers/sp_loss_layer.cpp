#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sp_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SpLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
}

template <typename Dtype>
void SpLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  const int num = bottom[0]->shape(0);
  channels_ = bottom[0]->shape(1);
  height_ = bottom[0]->shape(2);
  width_ = bottom[0]->shape(3);
  CHECK_EQ(height_, bottom[2]->shape(2));
  CHECK_EQ(width_, bottom[2]->shape(3));
  valid_vec_.resize(num, 0);
  max_idx_vec_.resize(num, 0);
}

template <typename Dtype>
Dtype SpLossLayer<Dtype>::get_max_idx(const Dtype* data, int len) {
    // find the largest index in the mask image
    Dtype max_idx = 0;
    for (int i = 0; i < len; ++i) {
      if (data[i] > max_idx)
        max_idx = data[i];
    }
    return max_idx;
}

template <typename Dtype>
void SpLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* mask_data = bottom[2]->cpu_data();
  int dim = prob_.count() / outer_num_;
  Dtype total_loss = 0.0;
  for (int i = 0; i < outer_num_; ++i) {
    Dtype loss = 0;
    const int max_idx = static_cast<int>(get_max_idx(mask_data, inner_num_) + 1);
    max_idx_vec_[i] = max_idx;
    
    // stat the information of each superpixel
    vector<int> sp_stat(max_idx * channels_, 0);
    vector<int> sp_size(max_idx, 0);
    for (int j = 0; j < inner_num_; ++j) {
      const int gt_label = static_cast<int>(label[j]);
      if (has_ignore_label_ && gt_label == ignore_label_) {
        continue;
      }
      const int idx_value = static_cast<int>(mask_data[j]);
      sp_size[idx_value] += 1;
      sp_stat[idx_value * channels_ + gt_label] += 1;
    }

    // compute the real label of each superpixel
    vector<int> sp_labels(max_idx, 0);
    for (int j = 0; j < max_idx; ++j) {
      if (sp_size[j] == 0)
        continue;
      int sp_max_label = 0;
      for (int c = 0; c < channels_; ++c) {
        const int sp_label = sp_stat[j * channels_ + c];
        if (sp_label > sp_max_label) {
          sp_max_label = sp_label;
          sp_labels[j] = c;
        }
      }
    }
    
    // compute the loss of each superpixel
    vector<Dtype> sp_prob(max_idx, 0.0);
    for (int j = 0; j < inner_num_; ++j) {
      const int gt_label = static_cast<int>(label[j]);
      if (has_ignore_label_ && gt_label == ignore_label_) {
        continue;
      }
      const int idx_value = static_cast<int>(mask_data[j]);
      sp_prob[idx_value] -= log(std::max(prob_data[sp_labels[idx_value] * inner_num_ + j], Dtype(FLT_MIN)));
    }
    
    // Jet down the number of valid superpixels and normalize each image
    int valid = 0;
    for (int j = 0; j < max_idx; ++j) {
      if (sp_size[j] > 0) {
        valid += 1;
        loss += (sp_prob[j] / sp_size[j]);
      }
    }
    loss /= valid; 
    total_loss += loss;
    label += inner_num_;
    mask_data += inner_num_;
    prob_data += dim;
  }
  
  // normalize on mini-batch
  top[0]->mutable_cpu_data()[0] = total_loss / outer_num_;
}

template <typename Dtype>
void SpLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1] || propagate_down[2]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* mask_data = bottom[2]->cpu_data();
    int dim = prob_.count() / outer_num_;
    for (int i = 0; i < outer_num_; ++i) {
      int max_idx = max_idx_vec_[i];
      vector<int> sp_stat(max_idx * channels_, 0);
      vector<int> sp_size(max_idx, 0);
      for (int j = 0; j < inner_num_; ++j) {
        const int gt_label = static_cast<int>(label[j]);
        if (has_ignore_label_ && gt_label == ignore_label_) {
          continue;
        }
        const int idx_value = static_cast<int>(mask_data[j]);
        sp_size[idx_value] += 1;
        sp_stat[idx_value * channels_ + gt_label] += 1;
      }

      vector<int> sp_labels(max_idx, 0);
      for (int j = 0; j < max_idx; ++j) {
        if (sp_size[j] == 0)
          continue;
        int sp_max_label = 0;
        for (int c = 0; c < channels_; ++c) {
          const int sp_label = sp_stat[j * channels_ + c];
          if (sp_label > sp_max_label) {
            sp_max_label = sp_label;
            sp_labels[j] = c;
          }
        }
      }

      vector<Dtype> sp_prob(max_idx, 0.0);
      for (int j = 0; j < inner_num_; ++j) {
        const int gt_label = static_cast<int>(label[j]);
        if (has_ignore_label_ && gt_label == ignore_label_) {
          for (int c = 0; c < channels_; ++c)
            bottom_diff[c * inner_num_ + j] = 0;
        } else {
          const int idx_value = static_cast<int>(mask_data[j]);
          const int sp_label = sp_labels[idx_value];
          bottom_diff[sp_label * inner_num_ + j] -= 1.0;
          for (int c = 0; c < channels_; ++c)
            bottom_diff[c * inner_num_ + j] /= sp_size[idx_value];
        }
      }
      int valid = 0;
      for (int j = 0; j < max_idx; ++j) {
        if (sp_size[j] > 0) {
          valid += 1;
        }
      }
      for (int j = 0; j < inner_num_; ++j) {
        for (int c = 0; c < channels_; ++c)
          bottom_diff[c * inner_num_ + j] /= valid;
      }
      label += inner_num_;
      mask_data += inner_num_;
      bottom_diff += dim;
    }
    
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / outer_num_;
    caffe_scal(bottom[0]->count(), loss_weight, bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(SpLossLayer);
#endif

INSTANTIATE_CLASS(SpLossLayer);
REGISTER_LAYER_CLASS(SpLoss);

}  // namespace caffe
