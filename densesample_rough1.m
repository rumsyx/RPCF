function [feature_map,height_lomo,width_lomo]=densesample_rough1(deep_feature,kernel_h,kernel_w,pad_h,pad_w)

%% 
 basis=permute(deep_feature,[2 1 3]);%% 
%% padding
basis= padarray(basis,[ pad_h,pad_w],0);
[deep_h1 deep_w1]=size(basis(:,:,1));
basis_horizon=basis;
tic;
for circ_x=1:(2/3*kernel_w)
 tmp1=circshift(basis,[0, -circ_x]);
 basis_horizon=max(basis_horizon, tmp1);
end
toc;
 basis_horizon= basis_horizon(1:deep_h1-2/3*kernel_h,1:deep_w1-2/3*kernel_w,:);
 [height_lomo,width_lomo]=size(basis_horizon(:,:,1));
basis_vertical=basis;

tic;
for circ_y=1:(2/3*kernel_h)
 tmp1=circshift(basis,[-circ_y, 0]);
 basis_vertical=max(basis_vertical, tmp1);
end
toc;
basis_vertical=basis_vertical(1:deep_h1-2/3*kernel_h,1:deep_w1-2/3*kernel_w,:);
feature_map=cat(3,basis_horizon,basis_vertical);

feature_map=permute(feature_map,[2 1 3]);












