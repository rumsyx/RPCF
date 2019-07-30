#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
/*

template <typename Dtype>   
__global__ void add_mask_H(const int n, Dtype* H_input, Dtype* patch_mask, Dtype* H_total, int num_per_channel, int num_per_sample) {
  CUDA_KERNEL_LOOP(index, n) {
  int current_index=index%num_per_channel;
  int current_index1=index%num_per_sample;
  H_total[index]=H_input[current_index1]*patch_mask[current_index];



  }
}

template <typename Dtype>   
__global__ void add_mask_second(const int n, int num_per_channel, Dtype* mask, float* input, float * output) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel;
   int current_index=index%num_per_channel;
   output[index]=input[index]*mask[current_index];

  }
}

void fft2_second(cufftHandle forward_plan, float* d_in, float2* d_freq)
{
    cufftExecR2C(forward_plan, d_in, d_freq);
    
}

void ifft2_second(cufftHandle inverse_plan, float2* d_freq, float* d_out)
{

     cufftExecC2R(inverse_plan, d_freq, d_out);
    
}

 template <typename Dtype>
__global__ void copy_freq(const int n, Dtype* real, Dtype* imag, float2* output) {
  CUDA_KERNEL_LOOP(index, n) {
  output[index].x=real[index];
  output[index].y=imag[index];
  }
}

template <typename Dtype>
__global__ void fftshift_second(const int n, int num_per_channel1, Dtype* L_mask, float2* input, Dtype* output_real, Dtype* output_imag) {
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

__global__ void scale_out_real_second(const int n, float* input, float scale_factor) {
  CUDA_KERNEL_LOOP(index, n) {
  input[index]=input[index]/scale_factor;
  }
}


template <typename Dtype>
__global__ void my_weight_sample_kernel1(const int n, Dtype* sample_real, Dtype* sample_imag,
    Dtype* weight_real, Dtype* weight_imag, Dtype* weighted_sample_real,Dtype* weighted_sample_imag, int number_per_sample,int number_per_channel) {
 CUDA_KERNEL_LOOP(index, n) { 
    int channel_num=number_per_sample/number_per_channel;
    int sample_index=index/number_per_channel;
    int position_index=index%number_per_channel;
    for(int i=0;i<channel_num;i++)
    {int hf_base_position=position_index+i*number_per_channel;

     weighted_sample_real[index]= weighted_sample_real[index]+weight_real[hf_base_position]*sample_real[hf_base_position+number_per_sample*sample_index]+weight_imag[hf_base_position]*sample_imag[hf_base_position+number_per_sample*sample_index];
    weighted_sample_imag[index]= weighted_sample_imag[index]-weight_real[hf_base_position]*sample_imag[hf_base_position+number_per_sample*sample_index]+weight_imag[hf_base_position]*sample_real[hf_base_position+number_per_sample*sample_index];
    } 

  }
}


template <typename Dtype>
__global__ void ifftshift_second(const int n, int num_per_channel, Dtype* L_mask, Dtype* input_real, Dtype* input_imag, float2* output, int row_num, int col_num,int num_per_channel1) {
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
__global__ void get_freq_second(const int n, float2* freq, Dtype* top_data_real) {
  CUDA_KERNEL_LOOP(index, n) {
  top_data_real[index]=freq[index].x; 

  }
}
*/

template <typename Dtype>
void WtfsecondLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    Dtype* data=Layer<Dtype>::feature_num[0]->mutable_cpu_data();
    int feature_num=data[0];

    int count1; int count2;  int count3;  int number_per_sample; float scale_factor; int col_num; int row_num; int num_per_channel1;
    int num_per_channel2;

    Dtype* sample_weight=Layer<Dtype>::sample_weight[0]->mutable_gpu_data();

    int sample_num=Layer<Dtype>::sample_weight[0]->width();
/*
//拷贝第一个卷积层代码到此处
Dtype lambda1=0.1; Dtype lambda2=1; Dtype lambda3=0.0;
Dtype* ifftshift_mask;Dtype* fftshift_mask; Dtype* weighted_sample_real;Dtype* weighted_sample_imag;
Dtype* sample_real; Dtype* sample_imag;
Dtype* KK_real;Dtype* KK_imag;
Dtype* tmp_real1;Dtype* tmp_imag1;
Dtype* hf_real;
Dtype* hf_imag; Dtype* laplace_real; Dtype* laplace_imag; Dtype* mask;

Dtype* data1=Layer<Dtype>::mu[0]->mutable_cpu_data();
Dtype* data2=Layer<Dtype>::eta[0]->mutable_cpu_data();
Dtype mu=data1[0]; Dtype eta=data2[0];

Dtype* frame_id_cpu=Layer<Dtype>::frame[0]->mutable_cpu_data();
int frame_id=frame_id_cpu[0];

for(int blob_id=0;blob_id<feature_num;blob_id++)
{
 if(blob_id!=2)
    {
      count1=Layer<Dtype>::KK_real[blob_id]->count();
      num_per_channel1=Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();  
       fftshift_mask=Layer<Dtype>::fftshift_mask[0]->mutable_gpu_data();
        row_num=Layer<Dtype>::KK_real[blob_id]->height(); col_num=row_num; 
        num_per_channel2=row_num*col_num;
        count2=num_per_channel2*this->blobs_[blob_id]->channels();
      ifftshift_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data(), this->d_freq2,row_num, col_num,num_per_channel1); 
         ifft2_second(this->inverse_plan[blob_id],this->d_freq2,this->d_in2);
         scale_factor=col_num*row_num; 
         scale_out_real_second<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in2,scale_factor);
         mask=Layer<Dtype>::binary_mask_adaptive[0]->mutable_gpu_data();
         add_mask_second<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in2, this->d_in_tmp2);

   //将当前输出,H_masked与ATAW_MC加权求和
        caffe_gpu_add1(count2,(Dtype*) this->d_in_tmp2,Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data(),(Dtype)1.0,mu, (Dtype*)this->d_in_tmp2); 
        caffe_gpu_add1(count2,(Dtype*) this->d_in_tmp2,Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data(),(Dtype)1.0,eta, (Dtype*)this->d_in_tmp2); 

        fft2_second(this->forward_plan[blob_id],this->d_in_tmp2,this->d_freq2);
         fftshift_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq2,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data());
 
     //将结果存入blob中
        caffe_copy(this->blobs_[blob_id]->count()/2,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),this->blobs_[blob_id]->mutable_gpu_data());
        caffe_copy(this->blobs_[blob_id]->count()/2,Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data(),this->blobs_[blob_id]->mutable_gpu_data()+this->blobs_[blob_id]->count()/2);
     // printf("%d %d %d %d\n",this->blobs_[blob_id]->num(),this->blobs_[blob_id]->channels(),this->blobs_[blob_id]->height(),this->blobs_[blob_id]->width());
          
    }
    else
    {
       count1=Layer<Dtype>::KK_real[blob_id]->count();
      num_per_channel1=Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      ifftshift_mask=Layer<Dtype>::ifftshift_mask[1]->mutable_gpu_data();  
       fftshift_mask=Layer<Dtype>::fftshift_mask[1]->mutable_gpu_data();
        row_num=Layer<Dtype>::KK_real[blob_id]->height(); col_num=row_num; 
        num_per_channel2=row_num*col_num;
        count2=num_per_channel2*this->blobs_[blob_id]->channels();
      ifftshift_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data(), this->d_freq3,row_num, col_num,num_per_channel1); 
         ifft2_second(this->inverse_plan[blob_id],this->d_freq3,this->d_in3);
         scale_factor=col_num*row_num; 
         scale_out_real_second<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in3,scale_factor);
         mask=Layer<Dtype>::binary_mask_adaptive[1]->mutable_gpu_data();
         add_mask_second<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in3, this->d_in_tmp3);

      //将当前输出,H_masked与ATAW_MC加权求和
        caffe_gpu_add1(count2,(Dtype*) this->d_in_tmp3,Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data(),(Dtype)1.0,(Dtype)0.5*mu, (Dtype*)this->d_in_tmp3); 
        caffe_gpu_add1(count2,(Dtype*) this->d_in_tmp3,Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data(),(Dtype)1.0,eta, (Dtype*)this->d_in_tmp3); 

        fft2_second(this->forward_plan[blob_id],this->d_in_tmp3,this->d_freq3);
         fftshift_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq3,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data());
     //  printf("the frame_id is %d\n",frame_id); 
      
       caffe_copy(this->blobs_[blob_id]->count()/2,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),this->blobs_[blob_id]->mutable_gpu_data());
       caffe_copy(this->blobs_[blob_id]->count()/2,Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data(),this->blobs_[blob_id]->mutable_gpu_data()+this->blobs_[blob_id]->count()/2);

    }

}
  
*/  
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
void WtfsecondLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(WtfsecondLayer);

}  // namespace caffe
