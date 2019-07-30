#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

 void fft2_fourth(cufftHandle forward_plan, float* d_in, float2* d_freq)
{
    cufftExecR2C(forward_plan, d_in, d_freq);
    
}

void ifft2_fourth(cufftHandle inverse_plan, float2* d_freq, float* d_out)
{

     cufftExecC2R(inverse_plan, d_freq, d_out);
    
}

template <typename Dtype>
__global__ void ifftshift_fourth(const int n, int num_per_channel, Dtype* L_mask, Dtype* input_real, Dtype* input_imag, float2* output, int row_num, int col_num,int num_per_channel1) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel1;
   int current_index=index%num_per_channel1;

   if(L_mask[current_index]>0) 
   {int ori_index=L_mask[current_index]-1+channel_index*num_per_channel1;
    output[index].x=input_real[ori_index];
    output[index].y=input_imag[ori_index];
   }
   else
   { int ori_index=-L_mask[current_index]-1+channel_index*num_per_channel1;
     output[index].x=input_real[ori_index];
     output[index].y=-input_imag[ori_index]; 
   }
  }
}

template <typename Dtype>
__global__ void fftshift_fourth(const int n, int num_per_channel1, Dtype* L_mask, float2* input, Dtype* output_real, Dtype* output_imag) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel1;
   int current_index=index%num_per_channel1;

   if(L_mask[current_index]>-0.5)
    {
      int ori_index=L_mask[current_index]+channel_index*num_per_channel1;
      output_real[index]=input[ori_index].x;
      output_imag[index]=input[ori_index].y;
     // output_real[index]=ori_index;
     // output_imag[index]=ori_index;     
    }
    else
    {
      int ori_index=-L_mask[current_index]+channel_index*num_per_channel1;
      output_real[index]=input[ori_index].x;
      output_imag[index]=-input[ori_index].y;//复数域求共轭操作
      //output_real[index]=ori_index;
      //output_imag[index]=-ori_index;//复数域求共轭操作  
    }
  }
} 

__global__ void scale_out_real_fourth(const int n, float* input, float scale_factor) {
  CUDA_KERNEL_LOOP(index, n) {
  input[index]=input[index]/scale_factor;
  }
}

template <typename Dtype>
__global__ void set_zeros_fourth(const int n, Dtype* in_out) {
  CUDA_KERNEL_LOOP(index, n) {
  in_out[index]=0;
  }
}

template <typename Dtype>
void WtffourthLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
    Dtype* data=Layer<Dtype>::feature_num[0]->mutable_cpu_data();
    int feature_num=data[0];
    //进行变量清零
    set_zeros_fourth<<<CAFFE_GET_BLOCKS(1), CAFFE_CUDA_NUM_THREADS>>>(1, top[0]->mutable_gpu_data()); 
  
    Dtype scale_factor; 
  
   for(int blob_id=0; blob_id<feature_num;blob_id++)
    {
      if(blob_id!=2)
     {
      int col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); 
      int row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); int num_per_channel1=row_num*(col_num/2+1);
      int num_per_channel2=row_num*col_num;  int count1=Layer<Dtype>::first_layer_hf_real[blob_id]->count();
      int count3=col_num*row_num*Layer<Dtype>::first_layer_hf_real[blob_id]->channels();
      Dtype* ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();
      Dtype* fftshift_mask=Layer<Dtype>::fftshift_mask[0]->mutable_gpu_data();   
      Dtype* xf_real=Layer<Dtype>::matlab_hf_real[blob_id]->mutable_gpu_data();
      Dtype* xf_imag=Layer<Dtype>::matlab_hf_imag[blob_id]->mutable_gpu_data();
      Dtype* yf_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
      Dtype* yf_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();  
      ifftshift_fourth<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, xf_real ,xf_imag, this->d_freq2,row_num, col_num,num_per_channel1); 
           ifft2_fourth(this->inverse_plan[blob_id],this->d_freq2,this->d_in2);
        scale_factor=col_num*row_num; 
         scale_out_real_fourth<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,this->d_in2,scale_factor);  
        ifftshift_fourth<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, yf_real , yf_imag, this->d_freq2,row_num, col_num,num_per_channel1); 
           ifft2_fourth(this->inverse_plan[blob_id],this->d_freq2,this->d_in_tmp2);
        scale_out_real_fourth<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,this->d_in_tmp2,scale_factor); 
        // top[0]->Reshape(1,3,row_num,col_num);
        // caffe_copy(top[0]->count(),(Dtype*)this->d_in_tmp2,top[0]->mutable_gpu_data());
         caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1,
         1,count3,
        (Dtype)row_num*col_num, (Dtype*)this->d_in2, (Dtype*)this->d_in_tmp2,
        (Dtype)1, top[0]->mutable_gpu_data());  
     }
      else
        {
            
      int col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); 
      int row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); int num_per_channel1=row_num*(col_num/2+1);
      int num_per_channel2=row_num*col_num;  int count1=Layer<Dtype>::first_layer_hf_real[blob_id]->count();
      int count3=col_num*row_num*Layer<Dtype>::first_layer_hf_real[blob_id]->channels();
      Dtype* ifftshift_mask=Layer<Dtype>::ifftshift_mask[1]->mutable_gpu_data();
      Dtype* fftshift_mask=Layer<Dtype>::fftshift_mask[1]->mutable_gpu_data();   
      Dtype* xf_real=Layer<Dtype>::matlab_hf_real[blob_id]->mutable_gpu_data();
      Dtype* xf_imag=Layer<Dtype>::matlab_hf_imag[blob_id]->mutable_gpu_data();
      Dtype* yf_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
      Dtype* yf_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();  
      ifftshift_fourth<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, xf_real , xf_imag, this->d_freq3,row_num, col_num,num_per_channel1); 
           ifft2_fourth(this->inverse_plan[blob_id],this->d_freq3,this->d_in3);
         scale_factor=col_num*row_num; 
         scale_out_real_fourth<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,this->d_in3,scale_factor);  
        ifftshift_fourth<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, yf_real , yf_imag, this->d_freq3,row_num, col_num,num_per_channel1); 
           ifft2_fourth(this->inverse_plan[blob_id],this->d_freq3,this->d_in_tmp3);
           scale_out_real_fourth<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,this->d_in_tmp3,scale_factor);   
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1,
        1,count3,
        (Dtype)row_num*col_num, (Dtype*)this->d_in3, (Dtype*)this->d_in_tmp3,
        (Dtype)1, top[0]->mutable_gpu_data());  

        }



  } 
  caffe_copy(top[0]->count(),top[0]->mutable_gpu_data(),Layer<Dtype>::inner_product_result[0]->mutable_gpu_data());
//至此已经计算得到pq
//Dtype* data1=Layer<Dtype>::rho[0]->mutable_cpu_data(); Dtype* data2=Layer<Dtype>::inner_product_result[0]->mutable_cpu_data();
//Dtype rho=data1[0]; Dtype pq=data2[0];




 Dtype* clear_memory_cpu=Layer<Dtype>::clear_memory[0]->mutable_cpu_data();
if(clear_memory_cpu[0]>0.5) //清空申请的memory
{
  cudaFree(this->d_in1); cudaFree(this->d_in2); cudaFree(this->d_in3); cudaFree(this->d_in4); 
 cudaFree(this->d_in_tmp1); cudaFree(this->d_in_tmp2); cudaFree(this->d_in_tmp3); cudaFree(this->d_in_tmp4); 
  cudaFree(this->d_freq1); cudaFree(this->d_freq2); cudaFree(this->d_freq3); cudaFree(this->d_freq4);
   cufftDestroy(this->forward_plan[0]); cufftDestroy(this->forward_plan[1]); cufftDestroy(this->forward_plan[2]); cufftDestroy(this->forward_plan[3]); 

    if(feature_num==5)
    {
      cudaFree(this->d_in5);
      cudaFree(this->d_in_tmp5);  
      cufftDestroy(this->forward_plan[4]);
      cufftDestroy(this->inverse_plan[4]);  
    }

} 


}   
 
    
  

template <typename Dtype>
void WtffourthLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO) << "start of convolutionlayer backward_gpu";
  //CHECK((this->kstride_h_ == 1) && (this->kstride_w_ == 1)) << "Backward_gpu is not implemented for fully convolutin."
 const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
 
const unsigned int* mask = this->mask_.gpu_data();

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    Dtype* top_diff_mutable = top[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    // Mask
  //  if (this->has_mask_ && this->phase_ == TRAIN) {
  //    const unsigned int* mask = this->mask_.gpu_data();
  //    for (int n = 0; n < this->num_; ++n) {
  //  this->backward_gpu_mask(top_diff_mutable + top[i]->offset(n), mask);
  //    }
  //  }
   

    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
	  if (this->kstride_h_ == 1) {
	    this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
	  } else {
	    this->fcn_weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
	    //LOG(INFO) << "fcn_weight_gpu_gemm";
	  }
        }
     // 我们对得到的weight_diff进行遮罩操作
     this->backward_gpu_mask(weight_diff, mask); 

        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
          
  //LOG(INFO) << "end of convolutionlayer backward_gpu";
}

INSTANTIATE_LAYER_GPU_FUNCS(WtffourthLayer);

}  // namespace caffe
