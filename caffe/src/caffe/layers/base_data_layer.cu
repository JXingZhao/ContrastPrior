#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

template <typename Dtype>
void BasePrefetchingLabelmapDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LabelmapBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  top[1]->ReshapeLike(batch->labelmap_);
  // Copy the labels.
  caffe_copy(batch->labelmap_.count(), batch->labelmap_.gpu_data(),
       top[1]->mutable_gpu_data());
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
 if(top.size() == 3)
  {
  top[2]->Reshape(top[1]->num(), 1, 1, 1);
  Dtype* exist_data = top[2]->mutable_cpu_data();
  const Dtype* label_data = top[1]->cpu_data();
  bool exist_sal = false;
  int offset = top[1]->offset(1);
  int offset_e = top[2]->offset(1);
  for (int n = 0; n < top[2]->num(); n++)
  {   
      for (int c = 0; c < top[1]->channels(); c++)
          for (int h = 0; h < top[1]->height(); h++)
              for (int w = 0; w < top[1]->width(); w++)
              {
                 const int index = c * top[1]->height() * top[1]->width() + h * top[1]->width() + w;
                 if(static_cast<float>(label_data[index]) > 0.0)
                     exist_sal = true;
              }
      if(exist_sal == false)
          exist_data[0] = static_cast<Dtype>(0);
      else 
          exist_data[0] = static_cast<Dtype>(1);
      //LOG(INFO)<<exist_data[0];
      label_data = label_data + offset;
      exist_data = exist_data + offset_e;
  }
  }

  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

template <typename Dtype>
void BasePrefetchingSuperpixelmapDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SuperpixelmapBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  top[1]->ReshapeLike(batch->labelmap_);
  // Copy the labels.
  caffe_copy(batch->labelmap_.count(), batch->labelmap_.gpu_data(),
       top[1]->mutable_gpu_data());
  top[2]->ReshapeLike(batch->superpixelmap_);
  caffe_copy(batch->superpixelmap_.count(), batch->superpixelmap_.gpu_data(),
       top[2]->mutable_gpu_data());
 if(top.size() == 4)
  {
  top[3]->Reshape(top[1]->num(), 1, 1, 1);
  Dtype* exist_data = top[3]->mutable_cpu_data();
  const Dtype* label_data = top[1]->cpu_data();
  bool exist_sal = false;
  int offset = top[1]->offset(1);
  int offset_e = top[3]->offset(1);
  for (int n = 0; n < top[1]->num(); n++)
  {   
      for (int c = 0; c < top[1]->channels(); c++)
          for (int h = 0; h < top[1]->height(); h++)
              for (int w = 0; w < top[1]->width(); w++)
              {
                 const int index = c * top[1]->height() * top[1]->width() + h * top[1]->width() + w;
                 if(static_cast<float>(label_data[index]) > 0.0)
                     exist_sal = true;
              }
      if(exist_sal == false)
          exist_data[0] = static_cast<Dtype>(0);
      else 
          exist_data[0] = static_cast<Dtype>(1);
      //LOG(INFO)<<exist_data[0];
      label_data = label_data + offset;
      exist_data = exist_data + offset_e;
  }
  } 

   // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

template <typename Dtype>
void ImageDimPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch =
    BasePrefetchingDataLayer<Dtype>::prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  if (output_data_dim_) {
    // Reshape to loaded labels.
    top[2]->ReshapeLike(batch->dim_);
    // Copy the labels.
    caffe_copy(batch->dim_.count(), batch->dim_.gpu_data(),
        top[2]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  BasePrefetchingDataLayer<Dtype>::prefetch_free_.push(batch);
}

template <typename Dtype>
void WeakSegPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  WeakSegBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(), top[0]->mutable_gpu_data());

  // Reshape to loaded labels.
  top[1]->ReshapeLike(batch->label_);
  // Copy the labels.
  caffe_copy(batch->label_.count(), batch->label_.gpu_data(), top[1]->mutable_gpu_data());

  if (output_sp_mask_) {
    // Reshape to loaded labels.
    top[2]->ReshapeLike(batch->sp_mask_);
    // Copy the labels.
    caffe_copy(batch->sp_mask_.count(), batch->sp_mask_.gpu_data(), top[2]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

template <typename Dtype>
void SoftLabelPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SoftBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->soft_label_);
    // Copy the labels.
    caffe_copy(batch->soft_label_.count(), batch->soft_label_.gpu_data(), top[1]->mutable_gpu_data());
    // Reshape to loaded labels.
    top[2]->ReshapeLike(batch->hard_label_);
    // Copy the labels.
    caffe_copy(batch->hard_label_.count(), batch->hard_label_.gpu_data(), top[2]->mutable_gpu_data());
    // Reshape to loaded labels.
    top[3]->ReshapeLike(batch->image_label_);
    // Copy the labels.
    caffe_copy(batch->image_label_.count(), batch->image_label_.gpu_data(), top[3]->mutable_gpu_data());
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingLabelmapDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingSuperpixelmapDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(ImageDimPrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(SoftLabelPrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(WeakSegPrefetchingDataLayer);

}  // namespace caffe
