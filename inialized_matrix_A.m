function [index_com, App]=inialized_matrix_A(filter_sz,binary_mask,pooling_step)

%%  Initialize matrix A index
index=1:(filter_sz(1,1)*filter_sz(1,2));
index=reshape(index,[filter_sz(1,1) filter_sz(1,2)]);
% index=index';
index_masked=index.*binary_mask;
[x_coe,y_coe]=find(index_masked);
height=max(x_coe)-min(x_coe)+1; width=max(y_coe)-min(y_coe)+1;
index_nonzero=nonzeros(index_masked);
index_nonzero=reshape(index_nonzero,[height, width]);
% pooling_step=2;
index_com=[];
if pooling_step~=1

for i=0:floor(height/pooling_step)
    for j=0:floor(width/pooling_step)
 
        batch=index_nonzero( i*pooling_step+1: min( (i+1)*pooling_step, height ),j*pooling_step+1: min( (j+1)*pooling_step,width)     );
  
            for p=1:length(batch(:))
                for q=p+1:length(batch(:))
                    index_com=[index_com; [batch(p) batch(q)]];
                end
            end
    end
end

App=zeros(size(index_com,1),filter_sz(1,1)*filter_sz(1,2) );
    for i=1:size(index_com,1)
         App(i,index_com(i,1))=1; App(i,index_com(i,2))=-1;
    end
else
    index_com=[]; App=[];
end