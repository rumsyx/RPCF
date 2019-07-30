function AW_MC=compute_AW_MC(A,H_MC,filter_sz)

channel_num=size(H_MC,3);
for channel_id=1:channel_num
    H=H_MC(:,:,channel_id);
    length1=size(A,1);
    AW=zeros(length1,1);
    for i=1:length1
         index1=A(i,1); index2=A(i,2);
         AW(i)=H(index1)-H(index2);
    end

    AW_MC( (channel_id-1)*length1+1: channel_id*length1,:)=AW;

end