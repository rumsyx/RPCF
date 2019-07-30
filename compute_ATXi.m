function ATXi=compute_ATXi(A,App,H_MC,Xi,filter_sz)
%% compute ATXi
length_total=numel(H_MC);
length_per_channel=filter_sz(1,1)*filter_sz(1,2);
ATXi=zeros(length_total,1);
channel_num=size(H_MC,3);
length1=size(A,1);
for channel_id=1:channel_num
     tmp=zeros(filter_sz(1,1)*filter_sz(1,2),1);
     Xi_tmp=Xi( (channel_id-1)*length1+1:channel_id*length1);
      for i=1:size(A,1)
          row1=A(i,1); row2=A(i,2);
          tmp(row1)=tmp(row1)+Xi_tmp(i);
          tmp(row2)=tmp(row2)-Xi_tmp(i);
      end
    ATXi( (channel_id-1)*length_per_channel+1:channel_id*length_per_channel)=tmp;
end
ATXi=reshape(ATXi,[filter_sz(1,1), filter_sz(1,2), channel_num]);
