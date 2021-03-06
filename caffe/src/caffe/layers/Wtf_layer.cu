#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>  
#include <cuda_runtime.h>
#include <cufft.h>
#include "common/inc/helper_functions.h"
#include "common/inc/helper_cuda.h"
typedef float2 Complex;
#define SIGNAL_SIZE        50
#define FILTER_KERNEL_SIZE 11

namespace caffe {

void fft2_second(cufftHandle forward_plan, float* d_in, float2* d_freq)
{
    cufftExecR2C(forward_plan, d_in, d_freq);
    
}

void ifft2_second(cufftHandle inverse_plan, float2* d_freq, float* d_out)
{

     cufftExecC2R(inverse_plan, d_freq, d_out);
    
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
__global__ void add_mask_second(const int n, int num_per_channel, Dtype* mask, float* input, float * output) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel;
   int current_index=index%num_per_channel;
   output[index]=input[index]*mask[current_index];

  }
}

template <typename Dtype>   
__global__ void add_reg_mask(const int n, int num_per_channel, Dtype* mask, Dtype* input, Dtype * output) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel;
   int current_index=index%num_per_channel;
   output[index]=input[index]*mask[current_index];

  }
}

void fft2(cufftHandle forward_plan, float* d_in, float2* d_freq)
{
    cufftExecR2C(forward_plan, d_in, d_freq);
    
}

void ifft2(cufftHandle inverse_plan, float2* d_freq, float* d_out)
{

     cufftExecC2R(inverse_plan, d_freq, d_out);
    
}

__global__ void copy_memory_to_blob(const int n, float2* mem1, float* tmp1, float* tmp2) {
  CUDA_KERNEL_LOOP(index, n) {
  }
}

__global__ void copy_memory_from_blob(const int n, float2* mem1, float* tmp1, float* tmp2) {
  CUDA_KERNEL_LOOP(index, n) {
    
  }
}   

template <typename Dtype>
__global__ void set_zeros(const int n, Dtype* in_out) {
  CUDA_KERNEL_LOOP(index, n) {
  in_out[index]=0;
  }
}

__global__ void scale_out_real(const int n, float* input, float scale_factor) {
  CUDA_KERNEL_LOOP(index, n) {
  input[index]=input[index]/scale_factor;
  }
}

template <typename Dtype>   
__global__ void add_mask(const int n, int num_per_channel, Dtype* mask, float* input, float * output) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel;
   int current_index=index%num_per_channel;
   output[index]=input[index]*mask[current_index];

  }
}

template <typename Dtype>
__global__ void ifftshift(const int n, int num_per_channel, Dtype* L_mask, Dtype* input_real, Dtype* input_imag, float2* output, int row_num, int col_num,int num_per_channel1) {
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
__global__ void fftshift(const int n, int num_per_channel1, Dtype* L_mask, float2* input, Dtype* output_real, Dtype* output_imag) {
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
__global__ void obtain_output(const int n,int number_per_sample1, int number_per_sample2,Dtype* L_mask, Dtype* real1, Dtype* real2, Dtype* real3, Dtype* real4, Dtype* real5,Dtype* imag1,Dtype* imag2, Dtype* imag3,Dtype* imag4,Dtype* imag5,Dtype* y_real, Dtype* y_imag) {
  CUDA_KERNEL_LOOP(index, n) {
 //我们首先判断当前的index是第几个样本的
    int sample_index1=index/number_per_sample1;
    int index1=index%number_per_sample1;
    int index2=number_per_sample2*sample_index1+L_mask[index1]-1;
   if(L_mask[index1]==0)
      {
        y_real[index]=real1[index]+real2[index]+real4[index]+real5[index];
        y_imag[index]=imag1[index]+imag2[index]+imag4[index]+imag5[index]; 
      }
   else
      { 
        y_real[index]=real1[index]+real2[index]+real3[index2]+real4[index]+real5[index];
        y_imag[index]=imag1[index]+imag2[index]+imag3[index2]+imag4[index]+imag5[index]; 

      }
  }
}

template <typename Dtype>
__global__ void obtain_freq(const int n, float2* input, Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
  output[index]=input[index].x;
  output[index+n]=input[index].y; 
  }
}   


template <typename Dtype>   
 __global__ void pad_filter(const int n,Dtype* pad_mask, int pad_h, int pad_w, int num_per_channel1, int num_per_channel2, int filter_h, int filter_w, int height, int width, int padded_height, int padded_width, Dtype* h_real_in, Dtype* h_imag_in , Dtype* h_real_out, Dtype* h_imag_out) {
 CUDA_KERNEL_LOOP(index, n) {
    // 首先确定当前的height和width
   int current_index=index%num_per_channel1;
   int channel_index=index/num_per_channel1;
   int index_ori=pad_mask[current_index]+channel_index*num_per_channel2;
   h_real_out[index]=h_real_in[0];
   h_imag_out[index]=h_imag_in[0]; 
  }
}

template <typename Dtype>   
__global__ void get_col(const int n, Dtype* col_mask, Dtype* h_real_in, Dtype* h_imag_in, Dtype* h_real_col, Dtype* h_imag_col) {
  CUDA_KERNEL_LOOP(index, n) {
  int index_ori=col_mask[index];
  h_real_col[index]=h_real_in[index_ori];
  h_imag_col[index]=h_imag_in[index_ori];
  

  }
}

template <typename Dtype>   
__global__ void get_freq(const int n, float2* freq, Dtype* top_data_real, Dtype* top_data_imag) {
  CUDA_KERNEL_LOOP(index, n) {
  top_data_real[index]=freq[index].x;
  top_data_imag[index]=freq[index].y; 

  }
}

template <typename Dtype>   
__global__ void set_freq(const int n, float2* freq, Dtype* input_data) {
  CUDA_KERNEL_LOOP(index, n) {
  freq[index].x=input_data[index];
  freq[index].y=input_data[index+n];

  }
}

template <typename Dtype>   
__global__ void laplace_add(const int n, Dtype* input1, Dtype* input2, Dtype* output1, Dtype* output2,Dtype factor) {
  CUDA_KERNEL_LOOP(index, n) {
  output1[index]=output1[index]+factor*input1[index];
  output2[index]=output2[index]+factor*input2[index];
  }
}


template <typename Dtype>
__global__ void my_weight_sample_kernel(const int n, Dtype* sample_real, Dtype* sample_imag,
    Dtype* weight_real, Dtype* weight_imag, Dtype* weighted_sample_real,Dtype* weighted_sample_imag, int number_per_sample,int number_per_channel) {
 CUDA_KERNEL_LOOP(index, n) { 
    int channel_num=number_per_sample/number_per_channel;
    int sample_index=index/number_per_channel;
    int position_index=index%number_per_channel;
    for(int i=0;i<channel_num;i++)
    {int hf_base_position=position_index+i*number_per_channel;
//weighted_sample_real[0]= weighted_sample_real[0]+weight_real[0]*sample_real[0]+weight_imag[0]*sample_imag[0];
// weighted_sample_real[1]=hf_base_position;    
    // weighted_sample_real[0]=sample_real[0];
  //   printf("the index is %d\n",index);
     weighted_sample_real[index]= weighted_sample_real[index]+weight_real[hf_base_position]*sample_real[hf_base_position+number_per_sample*sample_index]+weight_imag[hf_base_position]*sample_imag[hf_base_position+number_per_sample*sample_index];
    weighted_sample_imag[index]= weighted_sample_imag[index]-weight_real[hf_base_position]*sample_imag[hf_base_position+number_per_sample*sample_index]+weight_imag[hf_base_position]*sample_real[hf_base_position+number_per_sample*sample_index];
    } 

  }
}

template <typename Dtype>
__global__ void weight_sample_kernel_second(const int n, Dtype* sample_real, Dtype* sample_imag,
    Dtype* weighted_sample_real, Dtype* weighted_sample_imag, Dtype* KK_real,Dtype* KK_imag,Dtype* sample_weight, int number_per_sample,int number_per_channel, int sample_num) {
  CUDA_KERNEL_LOOP(index, n) {
 
  int position_index=index%number_per_channel;

    for(int i=0; i<sample_num;i++)
     {
        int weighted_sample_index=position_index+i*number_per_channel;
        int index1=index+i*number_per_sample;
        KK_real[index]=KK_real[index]+sample_weight[i]*(weighted_sample_real[weighted_sample_index]*sample_real[index1]-weighted_sample_imag[weighted_sample_index]*sample_imag[index1]);
        KK_imag[index]=KK_imag[index]+sample_weight[i]*(weighted_sample_real[weighted_sample_index]*sample_imag[index1]+weighted_sample_imag[weighted_sample_index]*sample_real[index1]);
      }



  }
}

template <typename Dtype>
__global__ void fuse_result(const int n, Dtype* input,Dtype* output, int channels, int num_per_channel2,int number_per_sample1 ) {
  CUDA_KERNEL_LOOP(index, n) {
 //首先判断当前元素是第几个frag
   for(int frag_id=0;frag_id<10;frag_id++)
    {  int position_index=index+number_per_sample1*frag_id;
         if(frag_id<9)
        {
          output[index]=output[index]+9*input[position_index];
        }
        else
        {
          output[index]=output[index]-input[position_index]; 
        }
    }

  }
}

template <typename Dtype>
__global__ void add_different_layers(const int n,int num_per_channel1, int num_per_channel2, Dtype* L_mask, Dtype* real,Dtype* imag, Dtype* sh_real, Dtype* sh_imag) {
  CUDA_KERNEL_LOOP(index, n) {
 //我们首先判断当前的index是第几个样本的
    int channel_index=index/num_per_channel1;
    int index1=index%num_per_channel1;
    int index2=num_per_channel2*channel_index+L_mask[index1]-1;
   if(L_mask[index1]==0)
      {
         sh_real[index]=sh_real[index]; 
         sh_imag[index]=sh_imag[index];
      }
   else
      { 
         sh_real[index]=sh_real[index]+real[index2];
         sh_imag[index]=sh_imag[index]+imag[index2];
      }
  }
}

template <typename Dtype>
__global__ void crop_sample(const int n,int num_per_channel1, int num_per_channel2, Dtype* L_mask1, Dtype* sh_real, Dtype* sh_imag, Dtype* output_real, Dtype* output_imag) {
  CUDA_KERNEL_LOOP(index, n) {
 //我们首先判断当前的index是第几个样本的
      int position_index=index%num_per_channel1;
      int channel_index=index/num_per_channel1;
      int index1=(L_mask1[position_index]-1)+num_per_channel2*channel_index;
      output_real[index]=sh_real[index1];
      output_imag[index]=sh_imag[index1];
  }
}

template <typename Dtype>
__global__ void compupte_H_transpose(const int n,Dtype* H, Dtype* H_transpose,  int num_per_channel_real, int height, int width) {
  CUDA_KERNEL_LOOP(index, n) {
  
  int channel_id=index/num_per_channel_real;
  int index1=index%num_per_channel_real;
  int height_id=index1/width;
  int width_id=index1%width;

  int new_index=width_id*width+height_id+channel_id*num_per_channel_real;
 
  H_transpose[new_index]=H[index];


}
}

template <typename Dtype>
__global__ void compute_AW(const int n, Dtype* H_transpose, Dtype* Ap, Dtype* AW, int A_height, int A_width, int num_per_channel_real) {
  CUDA_KERNEL_LOOP(index, n) {
  int channel_id=index/A_height;
  int current_index=index%A_height;    
 int index1=Ap[A_width*current_index]; int index2=Ap[A_width*current_index+1];
 AW[index]=H_transpose[index1+channel_id*num_per_channel_real]-H_transpose[index2+channel_id*num_per_channel_real]; 
}
}

template <typename Dtype>
__global__ void compute_ATAW_positive(const int n, Dtype* ATAW_positive_index, Dtype* ATAW, Dtype* AW, int AW_length_per_channel, int index_height, int index_width, int num_per_channel_real) {
  CUDA_KERNEL_LOOP(index, n) {
int channel_id=index/index_height;
int current_index=index%index_height;
int H_index=ATAW_positive_index[current_index*index_width]+channel_id*num_per_channel_real;
int AW_base_index=AW_length_per_channel*channel_id;
 for (int i=1;i<index_width;i++)
 {
   int AW_index=ATAW_positive_index[current_index*index_width+i];
    
   if(AW_index==-1)
    break;

   AW_index=AW_index+AW_base_index;
   ATAW[H_index]=ATAW[H_index]+AW[AW_index];
 }


}
}

template <typename Dtype>
__global__ void compute_ATAW_negative(const int n, Dtype* ATAW_negative_index, Dtype* ATAW, Dtype* AW, int AW_length_per_channel, int index_height, int index_width, int num_per_channel_real) {
  CUDA_KERNEL_LOOP(index, n) {
int channel_id=index/index_height;
int current_index=index%index_height;
int H_index=ATAW_negative_index[current_index*index_width]+channel_id*num_per_channel_real;
int AW_base_index=AW_length_per_channel*channel_id;
 for (int i=1;i<index_width;i++)
 {
   int AW_index=ATAW_negative_index[current_index*index_width+i];
    
   if(AW_index==-1)
    break;

   AW_index=AW_index+AW_base_index;
   ATAW[H_index]=ATAW[H_index]-AW[AW_index];
 }


}
}

template <typename Dtype>
__global__ void get_middle_line(const int n, Dtype* weight_sample, Dtype* middle_line, int height, int width) {
  CUDA_KERNEL_LOOP(index, n) {
  


  }
}

template <typename Dtype>
void WtfLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    Dtype* data=Layer<Dtype>::feature_num[0]->mutable_cpu_data();
    int feature_num=data[0];

    int count1; int count2;  int count3;  int number_per_sample; float scale_factor; int col_num; int row_num; int num_per_channel1;
    int num_per_channel2;

    Dtype* sample_weight=Layer<Dtype>::sample_weight[0]->mutable_gpu_data();
    Dtype* sample_weight_cpu=Layer<Dtype>::sample_weight[0]->mutable_cpu_data();
   

    int sample_num=Layer<Dtype>::sample_weight[0]->width();

    Dtype* index_cpu=Layer<Dtype>::index[0]->mutable_cpu_data();
    Dtype* index_cpu1=Layer<Dtype>::index1[0]->mutable_cpu_data();
    int index[feature_num];
    int index1[feature_num];
    for(int i=0;i<feature_num;i++)
    {
       index[i]=index_cpu[i];
       index1[i]=index_cpu1[i]; 
    }


    Dtype* ifftshift_mask;Dtype* fftshift_mask; Dtype* weighted_sample_real;Dtype* weighted_sample_imag;
    Dtype* sample_real; Dtype* sample_imag;
    Dtype* KK_real;Dtype* KK_imag;
    Dtype* tmp_real1;Dtype* tmp_imag1;
    Dtype* hf_real;
    Dtype* hf_imag; Dtype* laplace_real; Dtype* laplace_imag; Dtype* mask;


//××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
//××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
//××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
   for(int blob_id=0;blob_id<feature_num; blob_id++)
    {     
    //  printf("the value of blob_id is %d\n\n",blob_id); 
      if(blob_id!=2)
      {  
         ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();
         fftshift_mask=Layer<Dtype>::fftshift_mask[0]->mutable_gpu_data(); 

         weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
         weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();

         sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
         sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();

         KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data();
         KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();

         tmp_real1=Layer<Dtype>::first_layer_tmp_real1[blob_id]->mutable_gpu_data();
         tmp_imag1=Layer<Dtype>::first_layer_tmp_imag1[blob_id]->mutable_gpu_data();
  
         hf_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
         hf_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();

        laplace_real=Layer<Dtype>::laplace_real[blob_id]->mutable_gpu_data();
        laplace_imag=Layer<Dtype>::laplace_imag[blob_id]->mutable_gpu_data();
         col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); num_per_channel1=row_num*(col_num/2+1);
         num_per_channel2=row_num*col_num; 
          
          count1=this->blobs_[blob_id]->channels()*row_num*(col_num/2+1);//只考虑一半变量的反fftshift
          count2=this->blobs_[blob_id]->channels()*row_num*col_num;  
          count3=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
          number_per_sample=this->blobs_[blob_id]->channels()*(col_num/2+1)*row_num;
          ifftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::matlab_hf_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::matlab_hf_imag[blob_id]->mutable_gpu_data(), this->d_freq2,row_num, col_num,num_per_channel1); 
         ifft2(this->inverse_plan[blob_id],this->d_freq2,this->d_in2);
         scale_factor=col_num*row_num; 
         scale_out_real<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in2,scale_factor);  

         //首先测试求得的d_in是对的
          // top[0]->Reshape(1,Layer<Dtype>::KK_real[blob_id]->channels(),77,77);

         //对laplace_real及laplace_imag进行清零
         set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_real);
         set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_imag);

        // top[0]->Reshape(2,Layer<Dtype>::first_layer_tmp_real1[blob_id]->channels(),Layer<Dtype>::first_layer_tmp_real1[blob_id]->height(),Layer<Dtype>::first_layer_tmp_real1[blob_id]->width()); 
        //      printf("the size is %d %d\n",Layer<Dtype>::patch_mask[0]->height(),Layer<Dtype>::patch_mask[0]->width());

             mask=Layer<Dtype>::binary_mask[0]->mutable_gpu_data();
             add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in2, this->d_in_tmp2);

          //将结果写入变量写入H_masked并读出
         caffe_copy(Layer<Dtype>::H_masked[blob_id]->count(),(Dtype*)this->d_in_tmp2,Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data()); 

             fft2(this->forward_plan[blob_id],this->d_in_tmp2,this->d_freq2);
             //接着计算hf与samplesf的内积操作
              
             fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq2,Layer<Dtype>::hf_tmp_real[blob_id]->mutable_gpu_data(),
             Layer<Dtype>::hf_tmp_imag[blob_id]->mutable_gpu_data());

 

             mask=Layer<Dtype>::binary_mask_adaptive[0]->mutable_gpu_data();
             add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in2, this->d_in_tmp2);
             fft2(this->forward_plan[blob_id],this->d_in_tmp2,this->d_freq2);
             //接着计算hf与samplesf的内积操作
              
             fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq2,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),
             Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data());

             
             set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_real);
             set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_imag); 

            my_weight_sample_kernel<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3, sample_real, sample_imag,hf_real, hf_imag, weighted_sample_real,weighted_sample_imag,number_per_sample, num_per_channel1); 
        }
        else
        { 
             ifftshift_mask=Layer<Dtype>::ifftshift_mask[1]->mutable_gpu_data();
             fftshift_mask=Layer<Dtype>::fftshift_mask[1]->mutable_gpu_data(); 

             weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
             weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();

             sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
             sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();

             KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data();
             KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();

             tmp_real1=Layer<Dtype>::first_layer_tmp_real1[blob_id]->mutable_gpu_data();
             tmp_imag1=Layer<Dtype>::first_layer_tmp_imag1[blob_id]->mutable_gpu_data();
  
             hf_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
             hf_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();

             laplace_real=Layer<Dtype>::laplace_real[blob_id]->mutable_gpu_data();
             laplace_imag=Layer<Dtype>::laplace_imag[blob_id]->mutable_gpu_data();

             col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); num_per_channel1=row_num*(col_num/2+1);
             num_per_channel2=row_num*col_num; 
          
             count1=this->blobs_[blob_id]->channels()*row_num*(col_num/2+1);//只考虑一半变量的反fftshift
             count2=this->blobs_[blob_id]->channels()*row_num*col_num;  
             count3=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
             number_per_sample=this->blobs_[blob_id]->channels()*(col_num/2+1)*row_num;
            ifftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::matlab_hf_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::matlab_hf_imag[blob_id]->mutable_gpu_data(), this->d_freq3,row_num, col_num,num_per_channel1); 
             ifft2(this->inverse_plan[blob_id],this->d_freq3,this->d_in3);
             scale_factor=col_num*row_num; 
             scale_out_real<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in3,scale_factor); 

            //首先测试求得的d_in是对的
            // top[0]->Reshape(1,Layer<Dtype>::KK_real[blob_id]->channels(),77,77);

            //对laplace_real及laplace_imag进行清零
             set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_real);
             set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_imag);

             mask=Layer<Dtype>::binary_mask[1]->mutable_gpu_data();
             add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in3, this->d_in_tmp3);
       
             //将结果写入变量写入H_masked并读出
             caffe_copy(Layer<Dtype>::H_masked[blob_id]->count(),(Dtype*)this->d_in_tmp3,Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data());

             fft2(this->forward_plan[blob_id],this->d_in_tmp3,this->d_freq3);
             //接着计算hf与samplesf的内积操作
              
             fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq3,Layer<Dtype>::hf_tmp_real[blob_id]->mutable_gpu_data(),
             Layer<Dtype>::hf_tmp_imag[blob_id]->mutable_gpu_data());

          //  top[0]->Reshape(2,Layer<Dtype>::first_layer_tmp_real1[blob_id]->channels(),Layer<Dtype>::first_layer_tmp_real1[blob_id]->height(),Layer<Dtype>::first_layer_tmp_real1[blob_id]->width()); 
                mask=Layer<Dtype>::binary_mask_adaptive[1]->mutable_gpu_data();
                add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in3, this->d_in_tmp3);
                fft2(this->forward_plan[blob_id],this->d_in_tmp3,this->d_freq3);
              
               fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq3,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),
               Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data());

               set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_real);
               set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_imag); 

            my_weight_sample_kernel<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3, sample_real, sample_imag,hf_real, hf_imag, weighted_sample_real,weighted_sample_imag,number_per_sample,num_per_channel1); 
        }
    }
/*
Dtype* inner_product_result;
Dtype* tmp;
inner_product_result=Layer<Dtype>::inner_product_result[0]->mutable_gpu_data();
//首先在此处计算weighted_sample_real和weighted_sample_image的内积
for(int blob_id=0;blob_id<feature_num;blob_id++)
{
    number_per_sample=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->height()*Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->width();

    for(int sample_id=0; sample_id<sample_num;sample_id++)
  {
      tmp=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data()+number_per_sample*sample_id;
   if(sample_id==0&&blob_id==0)
   {
       caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, 1, number_per_sample,
                         (Dtype)2*sample_weight_cpu[sample_id]  , tmp, tmp,  (Dtype)0., inner_product_result);  
    }
    else
      {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, 1, number_per_sample,
                         (Dtype)2*sample_weight_cpu[sample_id], tmp, tmp, (Dtype)1, inner_product_result);   
   
      }
    //接着加虚部
     tmp=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data()+number_per_sample*sample_id; 
     caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, 1, number_per_sample,
                         (Dtype)2*sample_weight_cpu[sample_id], tmp, tmp, (Dtype)1, inner_product_result);   

   }
}
*/


   Dtype* L_index=Layer<Dtype>::L_index[0]->mutable_gpu_data(); 
   set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::sh_real[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::sh_real[0]->count(),Layer<Dtype>::sh_real[0]->mutable_gpu_data());
   set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::sh_imag[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::sh_imag[0]->count(),Layer<Dtype>::sh_imag[0]->mutable_gpu_data());
   Dtype* sh_real=Layer<Dtype>::sh_real[0]->mutable_gpu_data();
   Dtype* sh_imag=Layer<Dtype>::sh_imag[0]->mutable_gpu_data();
    
   for(int blob_id=0;blob_id<feature_num;blob_id++)
   {
       if(blob_id!=2)
       {
           caffe_gpu_add(Layer<Dtype>::sh_real[0]->count(),sh_real,Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data(),sh_real);
           caffe_gpu_add(Layer<Dtype>::sh_imag[0]->count(),sh_imag,Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data(),sh_imag); 
       }
       else
       {
        int count7=Layer<Dtype>::first_layer_weighted_sample_real[0]->count();
           num_per_channel1=Layer<Dtype>::first_layer_hf_real[0]->height()*(Layer<Dtype>::first_layer_hf_real[0]->width());
           num_per_channel2=Layer<Dtype>::first_layer_hf_real[2]->height()*(Layer<Dtype>::first_layer_hf_real[2]->width());
           //printf("the value is %d %d\n\n",num_per_channel1,num_per_channel2);
           add_different_layers<<<CAFFE_GET_BLOCKS(count7), CAFFE_CUDA_NUM_THREADS>>>(count7,num_per_channel1, num_per_channel2, L_index, Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data(), sh_real, sh_imag);
       }

   }
   
//接着利用sh_real及sh_imag接着计算输出
Dtype* L_index1=Layer<Dtype>::L_index1[0]->mutable_gpu_data();
  for(int blob_id=0;blob_id<feature_num;blob_id++)
  {
     if(blob_id!=2)
     {
      count1=this->blobs_[blob_id]->channels()*Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      num_per_channel1=Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      number_per_sample=num_per_channel1*this->blobs_[blob_id]->channels();
      KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data(); KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();
      sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
      sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();

      set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::KK_real[blob_id]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::KK_real[blob_id]->count(),KK_real);
      set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::KK_real[blob_id]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::KK_real[blob_id]->count(),KK_imag);

     weight_sample_kernel_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, sample_real, sample_imag, sh_real, sh_imag, KK_real,KK_imag,
                                                                                              sample_weight, number_per_sample,num_per_channel1, sample_num); 
     }
    else
    {
      count1=this->blobs_[blob_id]->channels()*Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      count2=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
      num_per_channel1=Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      num_per_channel2=Layer<Dtype>::first_layer_hf_real[0]->height()*Layer<Dtype>::first_layer_hf_real[0]->width();  
      number_per_sample=num_per_channel1*this->blobs_[blob_id]->channels();

      weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
      weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();


      KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data(); KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();

      set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::KK_real[blob_id]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::KK_real[blob_id]->count(),KK_real);
      set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::KK_real[blob_id]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::KK_real[blob_id]->count(),KK_imag);

      sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
      sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();
      crop_sample<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,num_per_channel1,num_per_channel2, L_index1, sh_real, sh_imag, Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data(), Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data());  
        //接着做第二次和样本的内积操作
     weight_sample_kernel_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, sample_real, sample_imag, weighted_sample_real, weighted_sample_imag, KK_real,KK_imag,sample_weight, number_per_sample,num_per_channel1, sample_num); 
    }
  }

int count_H;
int num_per_channel_real;
Dtype* App;
//计算ATAW_MC
Dtype* resolution_data=Layer<Dtype>::resolution_index[0]->mutable_cpu_data();
int resolution_index=resolution_data[0];
Dtype* ATAW_positive_index;
Dtype* ATAW_negative_index;


if(resolution_index==1)
{
  //printf("we select this branch\n");
  for(int blob_id=0; blob_id<feature_num;blob_id++)
  {
  
      count_H=Layer<Dtype>::H_transpose[blob_id]->count();
      num_per_channel_real=Layer<Dtype>::H_transpose[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->width();
      Dtype* H_masked=Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data();
      Dtype* H_transpose=Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data();

      compupte_H_transpose<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data(), Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data(), 
      num_per_channel_real, Layer<Dtype>::H_masked[blob_id]->height(), Layer<Dtype>::H_masked[blob_id]->width());

      count_H=Layer<Dtype>::Ap[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->channels();    
      compute_AW<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, H_transpose, Layer<Dtype>::Ap[blob_id]->mutable_gpu_data(), Layer<Dtype>::AW[blob_id]->mutable_gpu_data(), Layer<Dtype>::Ap[blob_id]->height(), Layer<Dtype>::Ap[blob_id]->width(), num_per_channel_real);
    //接着我们要去计算ATAW_MC
     ATAW_positive_index=Layer<Dtype>::ATAW_positive_index[blob_id]->mutable_gpu_data();
     ATAW_negative_index=Layer<Dtype>::ATAW_negative_index[blob_id]->mutable_gpu_data();

     

     //if(blob_id==1)
     // {
     //    top[0]->Reshape(1,1,Layer<Dtype>::AW[blob_id]->width(),1);
     //     caffe_copy(top[0]->count(),Layer<Dtype>::AW[blob_id]->mutable_gpu_data(),top[0]->mutable_gpu_data());
     // }
 
//首先对要计算的变量清零
      count_H=Layer<Dtype>::ATAW_MC[blob_id]->height()*Layer<Dtype>::ATAW_MC[blob_id]->width()*Layer<Dtype>::ATAW_MC[blob_id]->channels();
     set_zeros<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data());

     count_H=Layer<Dtype>::ATAW_positive_index[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->channels();
    compute_ATAW_positive<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, ATAW_positive_index, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data(), Layer<Dtype>::AW[blob_id]->mutable_gpu_data() , Layer<Dtype>::Ap[blob_id]->height(), Layer<Dtype>::ATAW_positive_index[blob_id]->height(), Layer<Dtype>::ATAW_positive_index[blob_id]->width(), num_per_channel_real);
count_H=Layer<Dtype>::ATAW_negative_index[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->channels();
compute_ATAW_negative<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, ATAW_negative_index, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data(), Layer<Dtype>::AW[blob_id]->mutable_gpu_data() , Layer<Dtype>::Ap[blob_id]->height(), Layer<Dtype>::ATAW_negative_index[blob_id]->height(), Layer<Dtype>::ATAW_negative_index[blob_id]->width(), num_per_channel_real);

    count_H=Layer<Dtype>::H_transpose[blob_id]->count();
    compupte_H_transpose<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data(), Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data(), 
      num_per_channel_real, Layer<Dtype>::H_masked[blob_id]->height(), Layer<Dtype>::H_masked[blob_id]->width());
   
     // if(blob_id==4)
     // {
     //     top[0]->Reshape(1,Layer<Dtype>::H_transpose[blob_id]->channels(),Layer<Dtype>::H_transpose[blob_id]->height(),Layer<Dtype>::H_transpose[blob_id]->width());
     //     caffe_copy(top[0]->count(),Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data(),top[0]->mutable_gpu_data());
     // }


   }
}
else
{
   //printf("we select this branch\n");
  for(int blob_id=0; blob_id<feature_num;blob_id++)
  {
      if(blob_id!=2)
      {
      count_H=Layer<Dtype>::H_transpose[blob_id]->count();
      num_per_channel_real=Layer<Dtype>::H_transpose[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->width();
      Dtype* H_masked=Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data();
      Dtype* H_transpose=Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data();

      compupte_H_transpose<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data(), Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data(), 
      num_per_channel_real, Layer<Dtype>::H_masked[blob_id]->height(), Layer<Dtype>::H_masked[blob_id]->width());

      count_H=Layer<Dtype>::Ap1[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->channels();    
      compute_AW<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, H_transpose, Layer<Dtype>::Ap1[blob_id]->mutable_gpu_data(), Layer<Dtype>::AW1[blob_id]->mutable_gpu_data(), Layer<Dtype>::Ap1[blob_id]->height(), Layer<Dtype>::Ap1[blob_id]->width(), num_per_channel_real);
    //接着我们要去计算ATAW_MC
     ATAW_positive_index=Layer<Dtype>::ATAW_positive_index1[blob_id]->mutable_gpu_data();
     ATAW_negative_index=Layer<Dtype>::ATAW_negative_index1[blob_id]->mutable_gpu_data();

//首先对要计算的变量清零
      count_H=Layer<Dtype>::ATAW_MC[blob_id]->height()*Layer<Dtype>::ATAW_MC[blob_id]->width()*Layer<Dtype>::ATAW_MC[blob_id]->channels();
     set_zeros<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data());

     count_H=Layer<Dtype>::ATAW_positive_index1[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->channels();
    compute_ATAW_positive<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, ATAW_positive_index, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data(), Layer<Dtype>::AW1[blob_id]->mutable_gpu_data() , Layer<Dtype>::Ap1[blob_id]->height(), Layer<Dtype>::ATAW_positive_index1[blob_id]->height(), Layer<Dtype>::ATAW_positive_index1[blob_id]->width(), num_per_channel_real);
count_H=Layer<Dtype>::ATAW_negative_index1[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->channels();
compute_ATAW_negative<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, ATAW_negative_index, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data(), Layer<Dtype>::AW1[blob_id]->mutable_gpu_data() , Layer<Dtype>::Ap1[blob_id]->height(), Layer<Dtype>::ATAW_negative_index1[blob_id]->height(), Layer<Dtype>::ATAW_negative_index1[blob_id]->width(), num_per_channel_real);

    count_H=Layer<Dtype>::H_transpose[blob_id]->count();
    compupte_H_transpose<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data(), Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data(), 
      num_per_channel_real, Layer<Dtype>::H_masked[blob_id]->height(), Layer<Dtype>::H_masked[blob_id]->width());

    //  if(blob_id==4)
    //  {
    //      top[0]->Reshape(1,Layer<Dtype>::H_transpose[blob_id]->channels(),Layer<Dtype>::H_transpose[blob_id]->height(),Layer<Dtype>::H_transpose[blob_id]->width());
    //      caffe_copy(top[0]->count(),Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data(),top[0]->mutable_gpu_data());
    //  }
     }
     else
     {
           count_H=Layer<Dtype>::H_transpose[blob_id]->count();
      num_per_channel_real=Layer<Dtype>::H_transpose[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->width();
      Dtype* H_masked=Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data();
      Dtype* H_transpose=Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data();

      compupte_H_transpose<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data(), Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data(), 
      num_per_channel_real, Layer<Dtype>::H_masked[blob_id]->height(), Layer<Dtype>::H_masked[blob_id]->width());

      count_H=Layer<Dtype>::Ap[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->channels();    
      compute_AW<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, H_transpose, Layer<Dtype>::Ap[blob_id]->mutable_gpu_data(), Layer<Dtype>::AW[blob_id]->mutable_gpu_data(), Layer<Dtype>::Ap[blob_id]->height(), Layer<Dtype>::Ap[blob_id]->width(), num_per_channel_real);
    //接着我们要去计算ATAW_MC
     ATAW_positive_index=Layer<Dtype>::ATAW_positive_index[blob_id]->mutable_gpu_data();
     ATAW_negative_index=Layer<Dtype>::ATAW_negative_index[blob_id]->mutable_gpu_data();

     

     //if(blob_id==1)
     // {
     //    top[0]->Reshape(1,1,Layer<Dtype>::AW[blob_id]->width(),1);
     //     caffe_copy(top[0]->count(),Layer<Dtype>::AW[blob_id]->mutable_gpu_data(),top[0]->mutable_gpu_data());
     // }
 
//首先对要计算的变量清零
      count_H=Layer<Dtype>::ATAW_MC[blob_id]->height()*Layer<Dtype>::ATAW_MC[blob_id]->width()*Layer<Dtype>::ATAW_MC[blob_id]->channels();
     set_zeros<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data());

     count_H=Layer<Dtype>::ATAW_positive_index[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->channels();
    compute_ATAW_positive<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, ATAW_positive_index, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data(), Layer<Dtype>::AW[blob_id]->mutable_gpu_data() , Layer<Dtype>::Ap[blob_id]->height(), Layer<Dtype>::ATAW_positive_index[blob_id]->height(), Layer<Dtype>::ATAW_positive_index[blob_id]->width(), num_per_channel_real);
count_H=Layer<Dtype>::ATAW_negative_index[blob_id]->height()*Layer<Dtype>::H_transpose[blob_id]->channels();
compute_ATAW_negative<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, ATAW_negative_index, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data(), Layer<Dtype>::AW[blob_id]->mutable_gpu_data() , Layer<Dtype>::Ap[blob_id]->height(), Layer<Dtype>::ATAW_negative_index[blob_id]->height(), Layer<Dtype>::ATAW_negative_index[blob_id]->width(), num_per_channel_real);

    count_H=Layer<Dtype>::H_transpose[blob_id]->count();
    compupte_H_transpose<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, Layer<Dtype>::ATAW_MC[blob_id]->mutable_gpu_data(), Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data(), 
      num_per_channel_real, Layer<Dtype>::H_masked[blob_id]->height(), Layer<Dtype>::H_masked[blob_id]->width());


     }



   }
}


//second layer加入到文件中

Dtype lambda1=0.1; Dtype lambda2=1; Dtype lambda3=0.0;

Dtype* data1=Layer<Dtype>::mu[0]->mutable_cpu_data();
Dtype* data2=Layer<Dtype>::eta[0]->mutable_cpu_data();
Dtype mu=data1[0]; Dtype eta=data2[0];

Dtype* data11=Layer<Dtype>::factor[0]->mutable_cpu_data();
Dtype factor=data11[0];

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

num_per_channel_real=Layer<Dtype>::H_masked[blob_id]->height()*Layer<Dtype>::H_masked[blob_id]->width();

count_H=Layer<Dtype>::H_masked[blob_id]->count();

mask=Layer<Dtype>::reg_window[0]->mutable_gpu_data();

        add_reg_mask<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, num_per_channel_real, mask, Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data(),Layer<Dtype>::H_reged[blob_id]->mutable_gpu_data());
   //将当前输出,H_masked与ATAW_MC加权求和
        caffe_gpu_add1(count2,(Dtype*) this->d_in_tmp2,Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data(),(Dtype)1.0,mu, (Dtype*)this->d_in_tmp2); 
        caffe_gpu_add1(count2,(Dtype*) this->d_in_tmp2,Layer<Dtype>::H_reged[blob_id]->mutable_gpu_data(),(Dtype)1.0,eta, (Dtype*)this->d_in_tmp2); 

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

num_per_channel_real=Layer<Dtype>::H_masked[blob_id]->height()*Layer<Dtype>::H_masked[blob_id]->width();

count_H=Layer<Dtype>::H_masked[blob_id]->count();

mask=Layer<Dtype>::reg_window[1]->mutable_gpu_data();

        add_reg_mask<<<CAFFE_GET_BLOCKS(count_H), CAFFE_CUDA_NUM_THREADS>>>(count_H, num_per_channel_real, mask, Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data(),Layer<Dtype>::H_reged[blob_id]->mutable_gpu_data());


      //将当前输出,H_masked与ATAW_MC加权求和
        caffe_gpu_add1(count2,(Dtype*)
            this->d_in_tmp3,Layer<Dtype>::H_transpose[blob_id]->mutable_gpu_data(),(Dtype)1.0,(Dtype)factor*mu, (Dtype*)this->d_in_tmp3); 
        caffe_gpu_add1(count2,(Dtype*) this->d_in_tmp3,Layer<Dtype>::H_masked[blob_id]->mutable_gpu_data(),(Dtype)1.0, (Dtype)0, (Dtype*)this->d_in_tmp3); 

        fft2_second(this->forward_plan[blob_id],this->d_in_tmp3,this->d_freq3);
         fftshift_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq3,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data());
     //  printf("the frame_id is %d\n",frame_id); 
      
       caffe_copy(this->blobs_[blob_id]->count()/2,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),this->blobs_[blob_id]->mutable_gpu_data());
       caffe_copy(this->blobs_[blob_id]->count()/2,Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data(),this->blobs_[blob_id]->mutable_gpu_data()+this->blobs_[blob_id]->count()/2);

    }

}

top[0]->Reshape(Layer<Dtype>::matlab_hf_real[0]->num(),Layer<Dtype>::matlab_hf_real[0]->channels(),Layer<Dtype>::matlab_hf_real[0]->height(),Layer<Dtype>::matlab_hf_real[0]->width());
//caffe_copy(top[0]->count(),Layer<Dtype>::matlab_hf_real[0]->mutable_gpu_data(),top[0]->mutable_gpu_data());
caffe_copy(top[0]->count(),Layer<Dtype>::first_layer_hf_real[0]->mutable_gpu_data(),top[0]->mutable_gpu_data());

Dtype* clear_memory_cpu=Layer<Dtype>::clear_memory[0]->mutable_cpu_data();
if(clear_memory_cpu[0]>0.5) //清空申请的memory
{
  cudaFree(this->d_in1); cudaFree(this->d_in2); cudaFree(this->d_in3); cudaFree(this->d_in4); 
 cudaFree(this->d_in_tmp1); cudaFree(this->d_in_tmp2); cudaFree(this->d_in_tmp3); cudaFree(this->d_in_tmp4); 
  cudaFree(this->d_freq1); cudaFree(this->d_freq2); cudaFree(this->d_freq3); cudaFree(this->d_freq4);
  cudaFree(this->d_in_total1); cudaFree(this->d_in_total2);
   cudaFree(this->d_freq_total1); cudaFree(this->d_freq_total2);
   cudaFree(this->d_in_sub_total1); cudaFree(this->d_in_sub_total2);
   cudaFree(this->d_freq_sub_total1); cudaFree(this->d_freq_sub_total2);

cufftDestroy(this->forward_plan[0]); cufftDestroy(this->forward_plan[1]); cufftDestroy(this->forward_plan[2]); cufftDestroy(this->forward_plan[3]);
 cufftDestroy(this->forward_plan_total[0]); cufftDestroy(this->forward_plan_total[1]);
 cufftDestroy(this->forward_plan_sub_total[0]); cufftDestroy(this->forward_plan_sub_total[1]);  

cufftDestroy(this->inverse_plan[0]); cufftDestroy(this->inverse_plan[1]); cufftDestroy(this->inverse_plan[2]); cufftDestroy(this->inverse_plan[3]);
 cufftDestroy(this->inverse_plan_total[0]); cufftDestroy(this->inverse_plan_total[1]); 


   if(feature_num==5)
    { printf("the memory is released\n");
      cudaFree(this->d_in5);
      cudaFree(this->d_in_tmp5);  
      cudaFree(this->d_freq5); 
      cufftDestroy(this->forward_plan[4]);
      cufftDestroy(this->inverse_plan[4]);  
    }


}



//Dtype* sample_real_cpu=Layer<Dtype>::first_layer_samplef_real[0]->mutable_cpu_data();
//Dtype* sample_imag_cpu=Layer<Dtype>::first_layer_samplef_imag[0]->mutable_cpu_data();
//Dtype* sh_real_cpu=Layer<Dtype>::sh_real[0]->mutable_cpu_data();
//Dtype* sh_imag_cpu=Layer<Dtype>::sh_imag[0]->mutable_cpu_data();


   //接着我们试着将weighted_sample_real加到一起
    //top[0]->Reshape(Layer<Dtype>::first_layer_weighted_sample_real[0]->num(),Layer<Dtype>::first_layer_weighted_sample_real[0]->channels(),Layer<Dtype>::first_layer_weighted_sample_real[0]->height(),
    //Layer<Dtype>::first_layer_weighted_sample_real[0]->width());
    //caffe_copy(top[0]->count(),sh_imag,top[0]->mutable_gpu_data());
    //top[0]->Reshape(Layer<Dtype>::KK_real[2]->num(),Layer<Dtype>::KK_real[2]->channels(),Layer<Dtype>::KK_real[2]->height(),Layer<Dtype>::KK_real[2]->width());
    //caffe_copy(top[0]->count(),Layer<Dtype>::KK_imag[2]->mutable_gpu_data(),top[0]->mutable_gpu_data());
 // top[0]->Reshape(Layer<Dtype>::first_layer_weighted_sample_real[2]->num(),Layer<Dtype>::first_layer_weighted_sample_real[2]->channels(),Layer<Dtype>::first_layer_weighted_sample_real[2]->height(),Layer<Dtype>::first_layer_weighted_sample_real[2]->width());
   //caffe_copy(top[0]->count(),Layer<Dtype>::first_layer_weighted_sample_real[2]->mutable_gpu_data(),top[0]->mutable_gpu_data()); 
   

}



template <typename Dtype>
void WtfLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {



}

INSTANTIATE_LAYER_GPU_FUNCS(WtfLayer);

}  // namespace caffe
