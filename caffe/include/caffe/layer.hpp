#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/device_alternate.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class mutex; }

namespace caffe {

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
template <typename Dtype>
class Layer {
 public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */
  explicit Layer(const LayerParameter& param)
    : layer_param_(param), is_shared_(false) {
      // Set phase and copy blobs (if there are any).
      phase_ = param.phase();
      if (layer_param_.blobs_size() > 0) {
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          blobs_[i].reset(new Blob<Dtype>());
          blobs_[i]->FromProto(layer_param_.blobs(i));
        }
      }
 
        //  input[0].reset(new Blob<Dtype>());

    
     input1.resize(30);
        for(int i=0;i<30;i++)
        input1[i]=new Blob<Dtype>();

    }
  virtual ~Layer() {}
    static vector<shared_ptr<Blob<Dtype> > > input;
    static vector<shared_ptr<Blob<Dtype> > > input2; //buff for the upper network
static    vector<Blob<Dtype>*> input1;
static vector<shared_ptr<Blob<Dtype> > > third_layer_col;
static vector<shared_ptr<Blob<Dtype> > > third_layer_template_col;
static vector<shared_ptr<Blob<Dtype> > > third_layer_masked_data;
static vector<shared_ptr<Blob<Dtype> > > third_layer_tmp_weight;
 static vector<shared_ptr<Blob<Dtype> > > third_layer_input_feature;
static vector<shared_ptr<Blob<Dtype> > > third_layer_tmp_diff;
static vector<shared_ptr<Blob<Dtype> > > third_layer_first_frame_feature;
static vector<shared_ptr<Blob<Dtype> > > rotation_tmp;
//为第五层定义中间变量    
static vector<shared_ptr<Blob<Dtype> > > fifthlayer_col_buff;
static vector<shared_ptr<Blob<Dtype> > > fifthlayer_tmp;
static vector<shared_ptr<Blob<Dtype> > > fifthlayer_template1;
static vector<shared_ptr<Blob<Dtype> > > fifthlayer_data_im;
static vector<shared_ptr<Blob<Dtype> > > fifthlayer_data;
static vector<shared_ptr<Blob<Dtype> > > sixthlayer_col_buff;
static vector<shared_ptr<Blob<Dtype> > > sixthlayer_col_buff1;
static vector<shared_ptr<Blob<Dtype> > > fifthlayer_tmp4;
static vector<shared_ptr<Blob<Dtype> > > fifthlayer_tmp5;
static vector<shared_ptr<Blob<Dtype> > > seventhlayer_template_x;
static vector<shared_ptr<Blob<Dtype> > > seventhlayer_template_x1;
static vector<shared_ptr<Blob<Dtype> > > seventhlayer_template_y;
static vector<shared_ptr<Blob<Dtype> > > seventhlayer_template_y1;
static vector<shared_ptr<Blob<Dtype> > > seventhlayer_tmp;
static vector<shared_ptr<Blob<Dtype> > > seventhlayer_col_buff;
static vector<shared_ptr<Blob<Dtype> > > seventhlayer_tmp1;
static vector<shared_ptr<Blob<Dtype> > > eighthlayer_tmp;
static vector<shared_ptr<Blob<Dtype> > > eighthlayer_col_buff;
static vector<shared_ptr<Blob<Dtype> > > eighthlayer_tmp1;

static vector<shared_ptr<Blob<Dtype> > > ninthlayer_tmp;
static vector<shared_ptr<Blob<Dtype> > > ninthlayer_col_buff;
static vector<shared_ptr<Blob<Dtype> > > ninthlayer_tmp1;
static vector<shared_ptr<Blob<Dtype> > > ninthlayer_template_tmp;
static vector<shared_ptr<Blob<Dtype> > > ninthlayer_col_buff1;
static vector<shared_ptr<Blob<Dtype> > > ninthlayer_tmp2;
static vector<shared_ptr<Blob<Dtype> > > ninthlayer_tmp3;
static vector<shared_ptr<Blob<Dtype> > > ninthlayer_tmp4;
static vector<shared_ptr<Blob<Dtype> > > momentum;

//新算法
static vector<shared_ptr<Blob<Dtype> > > first_layer_fft_real;
static vector<shared_ptr<Blob<Dtype> > > first_layer_fft_imag;

static vector<shared_ptr<Blob<Dtype> > > second_layer_fft_real;
static vector<shared_ptr<Blob<Dtype> > > second_layer_fft_imag;

static vector<shared_ptr<Blob<Dtype> > > neta_out_fft_real;
static vector<shared_ptr<Blob<Dtype> > > neta_out_fft_imag;
static vector<shared_ptr<Blob<Dtype> > > neta_loss_fft_real;
static vector<shared_ptr<Blob<Dtype> > > neta_loss_fft_imag;

static vector<shared_ptr<Blob<Dtype> > > first_layer_hf_real;
static vector<shared_ptr<Blob<Dtype> > > first_layer_hf_imag;
static vector<shared_ptr<Blob<Dtype> > > first_layer_samplef_real;
static vector<shared_ptr<Blob<Dtype> > > first_layer_samplef_imag;
static vector<shared_ptr<Blob<Dtype> > > first_layer_weighted_sample_real;
static vector<shared_ptr<Blob<Dtype> > > first_layer_weighted_sample_imag;
static vector<shared_ptr<Blob<Dtype> > > first_layer_weighted_sample_real1;
static vector<shared_ptr<Blob<Dtype> > > first_layer_weighted_sample_imag1;
static vector<shared_ptr<Blob<Dtype> > > first_layer_yf_real;
static vector<shared_ptr<Blob<Dtype> > > first_layer_yf_imag;
static vector<shared_ptr<Blob<Dtype> > > first_layer_yf_real1;
static vector<shared_ptr<Blob<Dtype> > > first_layer_yf_imag1;
static vector<shared_ptr<Blob<Dtype> > > first_layer_yf_diff_real;
static vector<shared_ptr<Blob<Dtype> > > first_layer_yf_diff_imag;
static vector<shared_ptr<Blob<Dtype> > > L_index;
static vector<shared_ptr<Blob<Dtype> > > sample_weight;
static vector<shared_ptr<Blob<Dtype> > > cropped_yf_real;
static vector<shared_ptr<Blob<Dtype> > > cropped_yf_imag;
static vector<shared_ptr<Blob<Dtype> > > reg_window;
static vector<shared_ptr<Blob<Dtype> > > fftshift_mask;
static vector<shared_ptr<Blob<Dtype> > > ifftshift_mask;
static vector<shared_ptr<Blob<Dtype> > > binary_mask;
static vector<shared_ptr<Blob<Dtype> > > patch_mask;
static vector<shared_ptr<Blob<Dtype> > > feature_num;
static vector<shared_ptr<Blob<Dtype> > > KK_real;
static vector<shared_ptr<Blob<Dtype> > > KK_imag;
static vector<shared_ptr<Blob<Dtype> > > first_layer_tmp_real1;
static vector<shared_ptr<Blob<Dtype> > > first_layer_tmp_imag1;
static vector<shared_ptr<Blob<Dtype> > > laplace_real;
static vector<shared_ptr<Blob<Dtype> > > laplace_imag;
static vector<shared_ptr<Blob<Dtype> > > filter_H;
static vector<shared_ptr<Blob<Dtype> > > H_total;
static vector<shared_ptr<Blob<Dtype> > > second_layer_hf_real;
static vector<shared_ptr<Blob<Dtype> > > second_layer_hf_imag;
static vector<shared_ptr<Blob<Dtype> > > second_layer_weighted_sample_real;
static vector<shared_ptr<Blob<Dtype> > > second_layer_weighted_sample_imag;
static vector<shared_ptr<Blob<Dtype> > > dt_height;
static vector<shared_ptr<Blob<Dtype> > > dt_width;
static vector<shared_ptr<Blob<Dtype> > > mask_out;
static vector<shared_ptr<Blob<Dtype> > > index;
static vector<shared_ptr<Blob<Dtype> > > index1;
static vector<shared_ptr<Blob<Dtype> > > total_num;
static vector<shared_ptr<Blob<Dtype> > > total_num1;
static vector<shared_ptr<Blob<Dtype> > > total_num2;
static vector<shared_ptr<Blob<Dtype> > > total_num3;
static vector<shared_ptr<Blob<Dtype> > > matlab_hf_real;
static vector<shared_ptr<Blob<Dtype> > > matlab_hf_imag;
static vector<shared_ptr<Blob<Dtype> > > sh_real;
static vector<shared_ptr<Blob<Dtype> > > sh_imag;
static vector<shared_ptr<Blob<Dtype> > > L_index1;
static vector<shared_ptr<Blob<Dtype> > > frame;
static vector<shared_ptr<Blob<Dtype> > > hf_tmp_real;
static vector<shared_ptr<Blob<Dtype> > > hf_tmp_imag;
static vector<shared_ptr<Blob<Dtype> > > binary_mask_adaptive;
static vector<shared_ptr<Blob<Dtype> > > input_xff_real;
static vector<shared_ptr<Blob<Dtype> > > input_xff_imag;
static vector<shared_ptr<Blob<Dtype> > > input_yff_real;
static vector<shared_ptr<Blob<Dtype> > > input_yff_imag;
static vector<shared_ptr<Blob<Dtype> > > inner_product;

static vector<shared_ptr<Blob<Dtype> > > H_masked;

static vector<shared_ptr<Blob<Dtype> > > clear_memory;

static vector<shared_ptr<Blob<Dtype> > > eta;
static vector<shared_ptr<Blob<Dtype> > > mu;

static vector<shared_ptr<Blob<Dtype> > > ATAW_MC;

static vector<shared_ptr<Blob<Dtype> > > Ap;
static vector<shared_ptr<Blob<Dtype> > > Ap1;
static vector<shared_ptr<Blob<Dtype> > > H_transpose;
static vector<shared_ptr<Blob<Dtype> > > AW;
static vector<shared_ptr<Blob<Dtype> > > AW1;
static vector<shared_ptr<Blob<Dtype> > > App;
static vector<shared_ptr<Blob<Dtype> > > App1;
static vector<shared_ptr<Blob<Dtype> > > resolution_index;
static vector<shared_ptr<Blob<Dtype> > > ATAW_positive_index;
static vector<shared_ptr<Blob<Dtype> > > ATAW_negative_index;
static vector<shared_ptr<Blob<Dtype> > > ATAW_positive_index1;
static vector<shared_ptr<Blob<Dtype> > > ATAW_negative_index1;
static vector<shared_ptr<Blob<Dtype> > > inner_product_result;
static vector<shared_ptr<Blob<Dtype> > > middle_line;
static vector<shared_ptr<Blob<Dtype> > > rho;
static vector<shared_ptr<Blob<Dtype> > > zelta;

static vector<shared_ptr<Blob<Dtype> > > H_reged;

static vector<shared_ptr<Blob<Dtype> > > PCA_feature;

static vector<shared_ptr<Blob<Dtype> > > factor;
    
//static vector<Blob<Dtype>* > input; 
    static void  get_input() {
           // input.resize(2);
           // input[0].reset(new Blob<Dtype>());
          }
  /**
   * @brief Implements common layer setup functionality.
   *
   * @param bottom the preshaped input blobs
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   *
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types,
   * followed by Reshape to set up sizes of top blobs and internal buffers.
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
   * This method may not be overridden.
   */
  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    InitMutex();
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);
  }

  /**
   * @brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   * @param top
   *     the allocated but unshaped output blobs
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   */
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  /**
   * @brief Whether a layer should be shared by multiple nets during data
   *        parallelism. By default, all layers except for data layers should
   *        not be shared. data layers should be shared to ensure each worker
   *        solver access data sequentially during data parallelism.
   */
  virtual inline bool ShareInParallel() const { return false; }

  /** @brief Return whether this layer is actually shared by other nets.
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then this function is expected return true.
   */
  inline bool IsShared() const { return is_shared_; }

  /** @brief Set whether this layer is actually shared by other nets
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then is_shared should be set true.
   */
  inline void SetShared(bool is_shared) {
    CHECK(ShareInParallel() || !is_shared)
        << type() << "Layer does not support sharing.";
    is_shared_ = is_shared;
  }

  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs.
   *
   * @param bottom the input blobs, with the requested input shapes
   * @param top the top blobs, which should be reshaped as needed
   *
   * This method should reshape top blobs as needed according to the shapes
   * of the bottom (input) blobs, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the layer can
   * accommodate the bottom blobs.
   */
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;

  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * \return The total loss from the layer.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   */
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Given the top blob error gradients, compute the bottom blob error
   *        gradients.
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  /**
   * @brief Returns the vector of learnable parameter blobs.
   */
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

  /**
   * @brief Returns the layer parameter.
   */
  const LayerParameter& layer_param() const { return layer_param_; }

  /**
   * @brief Writes the layer parameter to a protocol buffer
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   */
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }

  /**
   * @brief Sets the loss associated with a top blob at a given index.
   */
  inline void set_loss(const int top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

  /**
   * @brief Returns the layer type.
   */
  virtual inline const char* type() const { return ""; }

  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
  virtual inline int MinBottomBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  virtual inline int MaxBottomBlobs() const { return -1; }
  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  virtual inline int ExactNumTopBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  virtual inline int MinTopBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  virtual inline int MaxTopBlobs() const { return -1; }
  /**
   * @brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  virtual inline bool AutoTopBlobs() const { return false; }

  /**
   * @brief Return whether to allow force_backward for a given bottom blob
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  /**
   * @brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  inline bool param_propagate_down(const int param_id) {
    return (param_propagate_down_.size() > param_id) ?
        param_propagate_down_[param_id] : false;
  }

  inline void set_phase(Phase phase) {
    phase_ = phase;
  }
  /**
   * @brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   */
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }

  // Functions for FNC2CNN and CNN2FCN
  virtual inline void set_kstride(int kstride) {};
  virtual inline void set_pad(int pad) {};
  virtual inline void set_stride(int stride) {};
  virtual inline int get_stride() {return 0;};
  virtual inline void update_is1x1() {};
  virtual inline void update_ext_stride() {};
  virtual inline void check_poolmethod(PoolingParameter_PoolMethod method) {};
  virtual inline int get_kernel_size() {return 0;};



 protected:
  /** The protobuf that stores the layer parameters */
  LayerParameter layer_param_;
  /** The phase: TRAIN or TEST */
  Phase phase_;
  /** The vector that stores the learnable parameters as a set of blobs. */
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  vector<shared_ptr<Blob<Dtype> > > template_;
  /** Vector indicating whether to compute the diff of each param blob. */
  vector<bool> param_propagate_down_;

  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
  vector<Dtype> loss_;

  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }

  /**
   * Called by the parent Layer's SetUp to check that the number of bottom
   * and top Blobs provided as input match the expected numbers specified by
   * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
   */
  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << type() << " Layer takes " << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << type() << " Layer takes at most " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size())
          << type() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  }

  /**
   * Called by SetUp to initialize the weights associated with any top blobs in
   * the loss function. Store non-zero loss weights in the diff blob.
   */
  inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
          "unspecified or specified once per top blob.";
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        const Dtype loss_weight = layer_param_.loss_weight(top_id);
        if (loss_weight == Dtype(0)) { continue; }
        this->set_loss(top_id, loss_weight);
        const int count = top[top_id]->count();
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
        caffe_set(count, loss_weight, loss_multiplier);
      }
    }
  }

 private:
  /** Whether this layer is actually shared by other nets*/
  bool is_shared_;

  /** The mutex for sequential forward if this layer is shared */
  shared_ptr<boost::mutex> forward_mutex_;

  /** Initialize forward_mutex_ */
  void InitMutex();
  /** Lock forward_mutex_ if this layer is shared */
  void Lock();
  /** Unlock forward_mutex_ if this layer is shared */
  void Unlock();

  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Lock during forward to ensure sequential forward
  Lock();
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Forward_cpu(bottom, top);
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:
    Forward_gpu(bottom, top);
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  Unlock();
  return loss;
}

template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Backward_cpu(top, propagate_down, bottom);
    break;
  case Caffe::GPU:
    Backward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

// Serialize LayerParameter to protocol buffer
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}
 
  template <typename Dtype>
 vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::input;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::input2;
 template <typename Dtype>
vector<Blob<Dtype>*> Layer<Dtype>:: input1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::third_layer_col;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::third_layer_template_col;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::third_layer_masked_data;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::third_layer_tmp_weight;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::third_layer_input_feature;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::third_layer_tmp_diff;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::third_layer_first_frame_feature;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::rotation_tmp;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::fifthlayer_col_buff;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::fifthlayer_tmp;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::fifthlayer_template1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::fifthlayer_data_im;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::fifthlayer_data;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::sixthlayer_col_buff;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::sixthlayer_col_buff1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::fifthlayer_tmp4;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::fifthlayer_tmp5;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::seventhlayer_template_x;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::seventhlayer_template_x1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::seventhlayer_template_y;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::seventhlayer_template_y1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::seventhlayer_tmp;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::seventhlayer_col_buff;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::seventhlayer_tmp1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::eighthlayer_tmp;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::eighthlayer_col_buff;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::eighthlayer_tmp1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ninthlayer_tmp;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ninthlayer_col_buff;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ninthlayer_tmp1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ninthlayer_template_tmp;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ninthlayer_col_buff1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ninthlayer_tmp2;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ninthlayer_tmp3;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ninthlayer_tmp4;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::momentum;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_fft_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_fft_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::second_layer_fft_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::second_layer_fft_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::neta_out_fft_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::neta_out_fft_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::neta_loss_fft_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::neta_loss_fft_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_hf_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_hf_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_samplef_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_samplef_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_weighted_sample_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_weighted_sample_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_weighted_sample_real1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_weighted_sample_imag1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_yf_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_yf_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_yf_real1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_yf_imag1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_yf_diff_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_yf_diff_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::L_index;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::sample_weight;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::cropped_yf_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::cropped_yf_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::reg_window;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::fftshift_mask;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ifftshift_mask;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::binary_mask;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::patch_mask;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::feature_num;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::KK_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::KK_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_tmp_real1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::first_layer_tmp_imag1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::laplace_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::laplace_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::filter_H;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::H_total;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::second_layer_hf_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::second_layer_hf_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::second_layer_weighted_sample_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::second_layer_weighted_sample_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::dt_height;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::dt_width;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::mask_out;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::index;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::index1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::total_num;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::total_num1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::total_num2;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::total_num3;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::matlab_hf_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::matlab_hf_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::sh_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::sh_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::L_index1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::frame;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::hf_tmp_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::hf_tmp_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::binary_mask_adaptive;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::input_xff_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::input_xff_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::input_yff_real;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::input_yff_imag;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::inner_product;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::clear_memory;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::H_masked;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::eta;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::mu;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ATAW_MC;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::Ap;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::Ap1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::H_transpose;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::AW;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::AW1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::App;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::App1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::resolution_index;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ATAW_positive_index;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ATAW_negative_index;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ATAW_positive_index1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::ATAW_negative_index1;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::inner_product_result;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::middle_line;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::rho;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::zelta;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::H_reged;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::PCA_feature;
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > >Layer<Dtype>::factor;
//input1[0].reset(new Blob<float>());
}  // namespace caffe

#endif  // CAFFE_LAYER_H_