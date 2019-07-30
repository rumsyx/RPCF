#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void WtffifthLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
 
       
}    
 
    
  

template <typename Dtype>
void WtffifthLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
          
  //LOG(INFO) << "end of convolutionlayer backward_gpu";
}

INSTANTIATE_LAYER_GPU_FUNCS(WtffifthLayer);

}  // namespace caffe
