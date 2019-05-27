
#ifndef CAFFE_IMAGE_LABELMAP_DATA_LAYER_HPP_
#define CAFFE_IMAGE_LABELMAP_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
struct data_pair
{
    string source_filename;
    string groundtruth_filename;
    string superpixel_filename;
};

template <typename Dtype>
class ImageSuperpixelmapDataLayer : public BasePrefetchingSuperpixelmapDataLayer<Dtype> {
 public:
  explicit ImageSuperpixelmapDataLayer(const LayerParameter& param)
      : BasePrefetchingSuperpixelmapDataLayer<Dtype>(param) {}
  virtual ~ImageSuperpixelmapDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageSuperpixelmapData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return -1; } //could be three if considering label

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(SuperpixelmapBatch<Dtype>* batch);
  vector<data_pair> lines_;
  int lines_id_;
  bool normalize_;
};

}

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
