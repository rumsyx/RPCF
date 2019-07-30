function ATAW=compute_ATAW(A,H,filter_sz)
%% compute ATAW
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

a=1;
