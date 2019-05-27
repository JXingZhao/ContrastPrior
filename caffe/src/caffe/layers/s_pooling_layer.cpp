#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/s_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
using std::min;
using std::max;

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   CHECK_EQ(bottom.size() , 2) << "Wrong number of bottom blobs";
   CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
}

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(4 == bottom[0]->num_axes() && bottom[1]->num_axes() == 4) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
 // CHECK_EQ(bottom[0]->shape(), bottom[1]->shape()) << "bottomes must have same shape";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[0]->num(), channels_, bottom[0]->height(),
      width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  pooled_width_ = width_;
  pooled_height_ = height_;
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, height_,
        width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, height_,
        width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* super_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    {
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    int num_label = 0;
        for (int h = 0; h < height_; ++h){
          for (int w = 0; w < width_; ++w){
            const int tmp = h * width_ + w;
            if (static_cast<int>(super_data[tmp]) > num_label)
               num_label = static_cast<int>(super_data[tmp]);
          }
        }

    vector<Dtype> vec_l(num_label + 10,static_cast<Dtype>(0));
    vector<Dtype> vec_i(num_label + 10, static_cast<Dtype>(0));
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h){
          for(int w = 0; w < width_; ++w){
          const int index = h * width_ + w;
          if (bottom_data[index] > vec_l[super_data[index]]){
              vec_l[super_data[index]] = bottom_data[index];
              vec_i[super_data[index]] = static_cast<Dtype>(index);
          }
          }
        }

        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int pool_index = ph * pooled_width_ + pw;
            top_data[pool_index] = vec_l[super_data[pool_index]];
            if (use_top_mask) {
              top_mask[pool_index] = vec_i[super_data[pool_index]];
             } else {
                mask[pool_index] = static_cast<int>(vec_i[super_data[pool_index]]);
                 }      
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
        for (int i = 0; i < vec_l.size(); ++i){
          vec_l[i] = static_cast<Dtype>(0);
          vec_i[i] = static_cast<Dtype>(0);
        }
      }
    }
    break;
    }
  case PoolingParameter_PoolMethod_AVE:{
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
        int num_label = 0;
        for (int h = 0; h < height_; ++h){
          for (int w = 0; w < width_; ++w){
            const int tmp = h * width_ + w;
            if (static_cast<int>(super_data[tmp]) > num_label)
               num_label = static_cast<int>(super_data[tmp]);
          }
        }
    vec_ave.resize(num_label + 10);
    for(int i = 0; i < vec_ave.size(); ++i)
    {
        vec_ave[i].clear();
    }
    vector<Dtype> vec_l(num_label + 10,static_cast<Dtype>(0));
    vector<Dtype> vec_num(num_label + 10, static_cast<Dtype>(0));
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h){
          for(int w = 0; w < width_; ++w){
          const int index = h * width_ + w;
          vec_l[super_data[index]] += bottom_data[index];
          vec_num[static_cast<int>(super_data[index])]++;
          vec_ave[static_cast<int>(super_data[index])].push_back(index); 
          }
        }
        for ( int i = 0; i < vec_l.size(); ++i){
          if(vec_num[i] >= 0)
              vec_l[i] = vec_l[i] / vec_num[i];
        }
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int pool_index = ph * pooled_width_ + pw;
            top_data[pool_index] = vec_l[super_data[pool_index]];    
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        for (int i = 0; i < vec_l.size(); ++i){
          vec_l[i] = static_cast<Dtype>(0);
          vec_num[i] = static_cast<Dtype>(0);
        }
      }
    }   
      break;}
  case PoolingParameter_PoolMethod_STOCHASTIC:{
    NOT_IMPLEMENTED;
    break;}
  default:{
    LOG(FATAL) << "Unknown pooling method.";
  }
}
}

template <typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* super_data = bottom[1]->cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
  {
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
           }
   for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
   }
    break;
 }
  case PoolingParameter_PoolMethod_AVE:
  {
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
                const int pool_index = ph * pooled_width_ + pw;
                for ( int i = 0; i < vec_ave[super_data[pool_index]].size(); ++i)
                {
                bottom_diff[vec_ave[super_data[pool_index]][i]] +=
                top_diff[pool_index] / vec_ave[super_data[pool_index]].size();
                }
              }
            }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  }
}
}



INSTANTIATE_CLASS(SuperpixelPoolingLayer);
REGISTER_LAYER_CLASS(SuperpixelPooling);

}  // namespace caffe
