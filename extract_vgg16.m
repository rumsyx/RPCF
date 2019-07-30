function [xl,xl1]=extract_vgg16(img_samples,fsolver,feature_input,feature_blob4,gparams)
          
              fsolver.net.set_input_dim([0, size(img_samples,4), 3, size(img_samples,1), size(img_samples,2)]);
              feature_input.set_data(single(img_samples));
%               tic;
              fsolver.net.forward_prefilled();
%               toc;
%               tic;
              x=permute(feature_blob4.get_data(),[2 1 3 4]);
              
              x1=permute(fsolver.net.blobs('conv2_2').get_data(),[2 1 3 4]);
              
             xl=bsxfun(@times, x, ((size(x,1)*size(x,2))^gparams.normalize_size * size(x,3)^gparams.normalize_dim ./ ...
        (sum(abs(reshape(x, [], 1, 1, size(x,4))).^gparams.normalize_power, 1) + eps)).^(1/gparams.normalize_power));
              
              x=x1;
              
              xl1=bsxfun(@times, x, ((size(x,1)*size(x,2))^gparams.normalize_size * size(x,3)^gparams.normalize_dim ./ ...
        (sum(abs(reshape(x, [], 1, 1, size(x,4))).^gparams.normalize_power, 1) + eps)).^(1/gparams.normalize_power));
