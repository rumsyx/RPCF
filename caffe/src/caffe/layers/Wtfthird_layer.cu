#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

//该函数用于实现PCG的功能

namespace caffe {

void fft2_third(cufftHandle forward_plan, float* d_in, float2* d_freq)
{
    cufftExecR2C(forward_plan, d_in, d_freq);
    
}

void ifft2_third(cufftHandle inverse_plan, float2* d_freq, float* d_out)
{

     cufftExecC2R(inverse_plan, d_freq, d_out);
    
}

template <typename Dtype>
__global__ void ifftshift_third(const int n, int num_per_channel, Dtype* L_mask, Dtype* input_real, Dtype* input_imag, float2* output, int row_num, int col_num,int num_per_channel1) {
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
__global__ void fftshift_third(const int n, int num_per_channel1, Dtype* L_mask, float2* input, Dtype* output_real, Dtype* output_imag) {
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


template <typename Dtype>
__global__ void set_zeros(const int n, Dtype* rhs_samplef_real, Dtype* rhs_samplef_imag) {
  CUDA_KERNEL_LOOP(index, n) {
 
    rhs_samplef_real[index]=0;
    rhs_samplef_imag[index]=0;

  }
}

template <typename Dtype>
__global__ void weight_samples(const int n, Dtype* rhs_samplef_real, Dtype* rhs_samplef_imag, Dtype* samplesf_real, Dtype* samplesf_imag, Dtype* sample_weight, int sample_num, 
                               int num_per_sample_imag) {
  CUDA_KERNEL_LOOP(index, n) {
    int index1;
      for(int sample_id=0; sample_id<sample_num; sample_id++)
      {
       index1=num_per_sample_imag*sample_id+index;
       rhs_samplef_real[index]=rhs_samplef_real[index]+sample_weight[sample_id]*samplesf_real[index1];
       rhs_samplef_imag[index]=rhs_samplef_imag[index]+sample_weight[sample_id]*samplesf_imag[index1];

      }

  }
}

template <typename Dtype>
__global__ void comput_xf_yf(const int n, Dtype* rhs_samplef_real, Dtype* rhs_samplef_imag, Dtype* yf_real, Dtype* yf_imag, int num_per_channel_imag) {
  CUDA_KERNEL_LOOP(index, n) {
      int index1=index%num_per_channel_imag;
       rhs_samplef_real[index]=rhs_samplef_real[index]*yf_real[index1]-rhs_samplef_imag[index]*yf_imag[index1];
       rhs_samplef_imag[index]=rhs_samplef_real[index]*yf_imag[index1]+rhs_samplef_imag[index]*yf_real[index1];

  }
}

template <typename Dtype>   
__global__ void add_mask_third(const int n, int num_per_channel, Dtype* mask, float* input, float * output) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel;
   int current_index=index%num_per_channel;
   output[index]=input[index]*mask[current_index];

  }
}


template <typename Dtype>
void WtfthirdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
    Dtype* data=Layer<Dtype>::feature_num[0]->mutable_cpu_data();
    int feature_num=data[0];
    Dtype* resolution_data=Layer<Dtype>::resolution_index[0]->mutable_cpu_data();
    int resolution_index=resolution_data[0];
    Dtype* sample_weight=Layer<Dtype>::sample_weight[0]->mutable_gpu_data();
    Dtype* sample_weight_cpu=Layer<Dtype>::sample_weight[0]->mutable_cpu_data();

    Dtype* rhs_samplef_real; Dtype* rhs_samplef_imag; Dtype* samplesf_real; Dtype* samplesf_imag;

    int col_num, row_num, channel_num, sample_num, num_per_sample_imag, num_per_channel_imag, num_per_channel_real;
  
    sample_num=Layer<Dtype>::first_layer_samplef_real[0]->num();

    Dtype* fftshift_mask; Dtype* ifftshift_mask; 

    Dtype* binary_mask;
          

    int count1, count2;
    
    printf("we get here1 %d\n",feature_num);  

    for(int blob_id=0; blob_id<feature_num;blob_id++)
    {
       if(blob_id!=2)
        {
         rhs_samplef_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
         rhs_samplef_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();
         count1=Layer<Dtype>::first_layer_hf_real[blob_id]->count();
         set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, rhs_samplef_real, rhs_samplef_imag); 
  
         samplesf_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
        
         samplesf_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();

         col_num=Layer<Dtype>::first_layer_samplef_real[blob_id]->width();

         row_num=Layer<Dtype>::first_layer_samplef_real[blob_id]->height();

         num_per_sample_imag=col_num*row_num*Layer<Dtype>::first_layer_hf_real[blob_id]->channels();

         weight_samples<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, rhs_samplef_real, rhs_samplef_imag, samplesf_real, samplesf_imag, sample_weight, sample_num, num_per_sample_imag);

         Dtype* yf_real=Layer<Dtype>::first_layer_yf_real[0]->mutable_gpu_data();
         Dtype* yf_imag=Layer<Dtype>::first_layer_yf_imag[0]->mutable_gpu_data();

         num_per_channel_imag=row_num*col_num;

        comput_xf_yf<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, rhs_samplef_real, rhs_samplef_imag, yf_real, yf_imag, num_per_channel_imag);
    
        ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();

         ifftshift_third<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel_imag, ifftshift_mask, Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data(), this->d_freq2,row_num, col_num,num_per_channel_imag);    

         ifft2_third(this->inverse_plan[blob_id],this->d_freq2,this->d_in2);

         binary_mask=Layer<Dtype>::binary_mask[0]->mutable_gpu_data();

         count2=row_num*row_num*Layer<Dtype>::first_layer_hf_real[blob_id]->channels();

         num_per_channel_real=row_num*row_num;

         add_mask_third<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel_real, binary_mask, this->d_in2, this->d_in_tmp2);         

         if(blob_id==0)
        {
            printf("we get here\n");   
         top[0]->Reshape(1,Layer<Dtype>::first_layer_hf_real[blob_id]->channels(),Layer<Dtype>::first_layer_hf_real[blob_id]->height(),Layer<Dtype>::first_layer_hf_real[blob_id]->width());
         caffe_copy(top[0]->count(),Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),top[0]->mutable_gpu_data());
        }  

        }
        else
        {


        }


    }
    
 













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
void WtfthirdLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO) << "start of convolutionlayer backward_gpu";
  //CHECK((this->kstride_h_ == 1) && (this->kstride_w_ == 1)) << "Backward_gpu is not implemented for fully convolutin."
          
  //LOG(INFO) << "end of convolutionlayer backward_gpu";
}

INSTANTIATE_LAYER_GPU_FUNCS(WtfthirdLayer);

}  // namespace caffe
