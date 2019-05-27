#include <vector>

#include "gtest/gtest.h"
#include "boost/scoped_ptr.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/s_pooling_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
using boost::scoped_ptr;
namespace caffe {

template <typename TypeParam>
class SuperpixelPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SuperpixelPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>(2,2,5,5)),
        blob_bottom_superpixel(new Blob<Dtype>(1,1,5,5)),
        blob_top_(new Blob<Dtype>(2,2,5,5)),
        blob_top_mask_(new Blob<Dtype>(2,2,5,5)) {
        FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_superpixel);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_superpixel);
    blob_top_vec_.push_back(blob_top_);

        }
 
  virtual ~SuperpixelPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_superpixel;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_superpixel;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  void TestForwardSquare()
  {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
        // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 3 8]
    //     [1 2 5 5 2]
    //     [2 4 9 8 7]
    //     [0 5 6 7 4]
    for (int i = 0; i < 25 * num * channels; i += 25) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 9;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1;
      blob_bottom_->mutable_cpu_data()[i +  8] = 3;
      blob_bottom_->mutable_cpu_data()[i +  9] = 8;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 5;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 2;
      blob_bottom_->mutable_cpu_data()[i + 16] = 4;
      blob_bottom_->mutable_cpu_data()[i + 17] = 9;
      blob_bottom_->mutable_cpu_data()[i + 18] = 8;
      blob_bottom_->mutable_cpu_data()[i + 19] = 7;
      blob_bottom_->mutable_cpu_data()[i + 20] = 0;
      blob_bottom_->mutable_cpu_data()[i + 21] = 5;
      blob_bottom_->mutable_cpu_data()[i + 22] = 6;
      blob_bottom_->mutable_cpu_data()[i + 23] = 7;
      blob_bottom_->mutable_cpu_data()[i + 24] = 4;

    }
       // Input: 
    // [1,1,2,2,3]
    // [1,2,2,2,3]
    // [1,1,4,3,3]
    // [4 4 4 5 3]
    // [4 5 5 5 5]
    blob_bottom_superpixel->mutable_cpu_data()[0] = 1;
    blob_bottom_superpixel->mutable_cpu_data()[1] = 1;
    blob_bottom_superpixel->mutable_cpu_data()[2] = 2;
    blob_bottom_superpixel->mutable_cpu_data()[3] = 2;
    blob_bottom_superpixel->mutable_cpu_data()[4] = 3;
    blob_bottom_superpixel->mutable_cpu_data()[5] = 1;
    blob_bottom_superpixel->mutable_cpu_data()[6] = 2;
    blob_bottom_superpixel->mutable_cpu_data()[7] = 2;
    blob_bottom_superpixel->mutable_cpu_data()[8] = 2;
    blob_bottom_superpixel->mutable_cpu_data()[9] = 3;
    blob_bottom_superpixel->mutable_cpu_data()[10] = 1;
    blob_bottom_superpixel->mutable_cpu_data()[11] = 1;
    blob_bottom_superpixel->mutable_cpu_data()[12] = 4;
    blob_bottom_superpixel->mutable_cpu_data()[13] = 3;
    blob_bottom_superpixel->mutable_cpu_data()[14] = 3;
    blob_bottom_superpixel->mutable_cpu_data()[15] = 4;
    blob_bottom_superpixel->mutable_cpu_data()[16] = 4;
    blob_bottom_superpixel->mutable_cpu_data()[17] = 4;
    blob_bottom_superpixel->mutable_cpu_data()[18] = 5;
    blob_bottom_superpixel->mutable_cpu_data()[19] = 3;
    blob_bottom_superpixel->mutable_cpu_data()[20] = 4;
    blob_bottom_superpixel->mutable_cpu_data()[21] = 5;
    blob_bottom_superpixel->mutable_cpu_data()[22] = 5;
    blob_bottom_superpixel->mutable_cpu_data()[23] = 5;
    blob_bottom_superpixel->mutable_cpu_data()[24] = 5;

    SuperpixelPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 5);
    EXPECT_EQ(blob_top_->width(), 5);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 5);
      EXPECT_EQ(blob_top_mask_->width(), 5);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Input:
    // [9,9,5,5,8]
    // [9,5,5,5,8]
    // [9,9,9,8,8]
    // [9,9,9,8,8]
    // [9,8,8,8,8]
    for (int i = 0; i < 25 * num * channels; i += 25) {
    EXPECT_EQ(blob_top_->cpu_data()[i + 0], 9);
    EXPECT_EQ(blob_top_->cpu_data()[i + 1], 9);
    EXPECT_EQ(blob_top_->cpu_data()[i + 2], 5);
    EXPECT_EQ(blob_top_->cpu_data()[i + 3], 5);
    EXPECT_EQ(blob_top_->cpu_data()[i + 4], 8);
    EXPECT_EQ(blob_top_->cpu_data()[i + 5], 9);
    EXPECT_EQ(blob_top_->cpu_data()[i + 6], 5);
    EXPECT_EQ(blob_top_->cpu_data()[i + 7], 5);
    EXPECT_EQ(blob_top_->cpu_data()[i + 8], 5);
    EXPECT_EQ(blob_top_->cpu_data()[i + 9], 8);
    EXPECT_EQ(blob_top_->cpu_data()[i + 10], 9);
    EXPECT_EQ(blob_top_->cpu_data()[i + 11], 9);
    EXPECT_EQ(blob_top_->cpu_data()[i + 12], 9);
    EXPECT_EQ(blob_top_->cpu_data()[i + 13], 8);
    EXPECT_EQ(blob_top_->cpu_data()[i + 14], 8);
    EXPECT_EQ(blob_top_->cpu_data()[i + 15], 9);
    EXPECT_EQ(blob_top_->cpu_data()[i + 16], 9);
    EXPECT_EQ(blob_top_->cpu_data()[i + 17], 9);
    EXPECT_EQ(blob_top_->cpu_data()[i + 18], 8);
    EXPECT_EQ(blob_top_->cpu_data()[i + 19], 8);
    EXPECT_EQ(blob_top_->cpu_data()[i + 20], 9);
    EXPECT_EQ(blob_top_->cpu_data()[i + 21], 8);
    EXPECT_EQ(blob_top_->cpu_data()[i + 22], 8);
    EXPECT_EQ(blob_top_->cpu_data()[i + 23], 8);
    EXPECT_EQ(blob_top_->cpu_data()[i + 24], 8);

    }
    if (blob_top_vec_.size() > 1) {
      // Expected mask output: 2x 2 channels of:
      //     [5  2  2 9]
      //     [5 12 12 9]
    EXPECT_EQ(blob_top_mask_->cpu_data()[0], 5);
    EXPECT_EQ(blob_top_mask_->cpu_data()[1], 5);
    EXPECT_EQ(blob_top_mask_->cpu_data()[2], 2);
    EXPECT_EQ(blob_top_mask_->cpu_data()[3], 2);
    EXPECT_EQ(blob_top_mask_->cpu_data()[4], 9);
    EXPECT_EQ(blob_top_mask_->cpu_data()[5], 5);
    EXPECT_EQ(blob_top_mask_->cpu_data()[6], 2);
    EXPECT_EQ(blob_top_mask_->cpu_data()[7], 2);
    EXPECT_EQ(blob_top_mask_->cpu_data()[8], 9);
    EXPECT_EQ(blob_top_mask_->cpu_data()[9], 9);
    EXPECT_EQ(blob_top_mask_->cpu_data()[10], 5);
    EXPECT_EQ(blob_top_mask_->cpu_data()[11], 5);
    EXPECT_EQ(blob_top_mask_->cpu_data()[12], 9);
    EXPECT_EQ(blob_top_mask_->cpu_data()[13], 9);
    EXPECT_EQ(blob_top_mask_->cpu_data()[14], 9);
    }
    LayerParameter layer_param_1;
    PoolingParameter* pooling_param_1 = layer_param_1.mutable_pooling_param();
    pooling_param_1->set_pool(PoolingParameter_PoolMethod_AVE);
    SuperpixelPoolingLayer<Dtype> layer_1(layer_param_1);
    layer_1.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 5);
    EXPECT_EQ(blob_top_->width(), 5);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 5);
      EXPECT_EQ(blob_top_mask_->width(), 5);
    }
    layer_1.Forward(blob_bottom_vec_, blob_top_vec_);
    for (int i = 0; i < 25 * num * channels; i += 25) {
    EXPECT_EQ(blob_top_->cpu_data()[i + 0], 3);
    EXPECT_EQ(blob_top_->cpu_data()[i + 1], 3);
    EXPECT_EQ(blob_top_->cpu_data()[i + 2], 3);
    EXPECT_EQ(blob_top_->cpu_data()[i + 3], 3);
    EXPECT_EQ(blob_top_->cpu_data()[i + 4], 5);
    EXPECT_EQ(blob_top_->cpu_data()[i + 5], 3);
    EXPECT_EQ(blob_top_->cpu_data()[i + 6], 3);
    EXPECT_EQ(blob_top_->cpu_data()[i + 7], 3);
    EXPECT_EQ(blob_top_->cpu_data()[i + 8], 3);
    EXPECT_EQ(blob_top_->cpu_data()[i + 9], 5);
    EXPECT_EQ(blob_top_->cpu_data()[i + 10], 3);
    EXPECT_EQ(blob_top_->cpu_data()[i + 11], 3);
    EXPECT_EQ(blob_top_->cpu_data()[i + 12], 4);
    EXPECT_EQ(blob_top_->cpu_data()[i + 13], 5);
    EXPECT_EQ(blob_top_->cpu_data()[i + 14], 5);
    EXPECT_EQ(blob_top_->cpu_data()[i + 15], 4);
    EXPECT_EQ(blob_top_->cpu_data()[i + 16], 4);
    EXPECT_EQ(blob_top_->cpu_data()[i + 17], 4);
    EXPECT_EQ(blob_top_->cpu_data()[i + 18], 6);
    EXPECT_EQ(blob_top_->cpu_data()[i + 19], 5);
    EXPECT_EQ(blob_top_->cpu_data()[i + 20], 4);
    EXPECT_EQ(blob_top_->cpu_data()[i + 21], 6);
    EXPECT_EQ(blob_top_->cpu_data()[i + 22], 6);
    EXPECT_EQ(blob_top_->cpu_data()[i + 23], 6);
    EXPECT_EQ(blob_top_->cpu_data()[i + 24], 6);

    }

  }
};

TYPED_TEST_CASE(SuperpixelPoolingLayerTest,TestDtypesAndDevices);
TYPED_TEST(SuperpixelPoolingLayerTest,TestForward)
{
    this->TestForwardSquare();
   }
TYPED_TEST(SuperpixelPoolingLayerTest, TestGradientMax) {
  typedef typename TypeParam::Dtype Dtype;
      LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
    const int num = 2;
    const int channels = 2;
        // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
     for (int i = 0; i < 25 * num * channels; i += 25) {
      this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      this->blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      this->blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      this->blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      this->blob_bottom_->mutable_cpu_data()[i +  5] = 9;
      this->blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      this->blob_bottom_->mutable_cpu_data()[i +  7] = 1;
      this->blob_bottom_->mutable_cpu_data()[i +  8] = 3;
      this->blob_bottom_->mutable_cpu_data()[i +  9] = 8;
      this->blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      this->blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      this->blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      this->blob_bottom_->mutable_cpu_data()[i + 13] = 5;
      this->blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      this->blob_bottom_->mutable_cpu_data()[i + 15] = 2;
      this->blob_bottom_->mutable_cpu_data()[i + 16] = 4;
      this->blob_bottom_->mutable_cpu_data()[i + 17] = 9;
      this->blob_bottom_->mutable_cpu_data()[i + 18] = 8;
      this->blob_bottom_->mutable_cpu_data()[i + 19] = 7;
      this->blob_bottom_->mutable_cpu_data()[i + 20] = 0;
      this->blob_bottom_->mutable_cpu_data()[i + 21] = 5;
      this->blob_bottom_->mutable_cpu_data()[i + 22] = 6;
      this->blob_bottom_->mutable_cpu_data()[i + 23] = 7;
      this->blob_bottom_->mutable_cpu_data()[i + 24] = 4;

    }
      // Input: 
    // [1,1,2,2,3]
    // [1,2,2,3,3]
    // [1,1,3,3,3]
    this->blob_bottom_superpixel->mutable_cpu_data()[0] = 1;
    this->blob_bottom_superpixel->mutable_cpu_data()[1] = 1;
    this->blob_bottom_superpixel->mutable_cpu_data()[2] = 2;
    this->blob_bottom_superpixel->mutable_cpu_data()[3] = 2;
    this->blob_bottom_superpixel->mutable_cpu_data()[4] = 3;
    this->blob_bottom_superpixel->mutable_cpu_data()[5] = 1;
    this->blob_bottom_superpixel->mutable_cpu_data()[6] = 2;
    this->blob_bottom_superpixel->mutable_cpu_data()[7] = 2;
    this->blob_bottom_superpixel->mutable_cpu_data()[8] = 2;
    this->blob_bottom_superpixel->mutable_cpu_data()[9] = 3;
    this->blob_bottom_superpixel->mutable_cpu_data()[10] = 1;
    this->blob_bottom_superpixel->mutable_cpu_data()[11] = 1;
    this->blob_bottom_superpixel->mutable_cpu_data()[12] = 4;
    this->blob_bottom_superpixel->mutable_cpu_data()[13] = 3;
    this->blob_bottom_superpixel->mutable_cpu_data()[14] = 3;
    this->blob_bottom_superpixel->mutable_cpu_data()[15] = 4;
    this->blob_bottom_superpixel->mutable_cpu_data()[16] = 4;
    this->blob_bottom_superpixel->mutable_cpu_data()[17] = 4;
    this->blob_bottom_superpixel->mutable_cpu_data()[18] = 5;
    this->blob_bottom_superpixel->mutable_cpu_data()[19] = 3;
    this->blob_bottom_superpixel->mutable_cpu_data()[20] = 4;
    this->blob_bottom_superpixel->mutable_cpu_data()[21] = 5;
    this->blob_bottom_superpixel->mutable_cpu_data()[22] = 5;
    this->blob_bottom_superpixel->mutable_cpu_data()[23] = 5;
    this->blob_bottom_superpixel->mutable_cpu_data()[24] = 5;
    SuperpixelPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_,this->blob_top_vec_);
      GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_, 0);
}
}
