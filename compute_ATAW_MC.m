function ATAW_MC=compute_ATAW_MC(A,App,H_MC,filter_sz)
%% compute ATAW in multiple channels

channel_num=size(H_MC,3);
for channel_id=1:channel_num
    H=H_MC(:,:,channel_id);
    length1=size(A,1);
    AW=zeros(length1,1);
    for i=1:length1
         index1=A(i,1); index2=A(i,2);
         AW(i)=H(index1)-H(index2);
    end
     %% compute index for AT
    length_total=length(H(:));

    ATAW=zeros(length_total,1);
    for i=1:size(A,1)
          row1=A(i,1); row2=A(i,2);
          ATAW(row1)=ATAW(row1)+AW(i);
          ATAW(row2)=ATAW(row2)-AW(i);
    end
    ATAW=reshape(ATAW,[filter_sz(1,1) filter_sz(1,2)]);
    ATAW_MC(:,:,channel_id)=ATAW;
end
a=1;
