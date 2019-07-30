function [pos,neg]=get_sample(map_part,x_final,y_final,dis_vec_total)

[height,width]=size(map_part(:,:,1));
feature_tmp=reshape(map_part,[height*width,100]);
 [index,position]=max(feature_tmp,[],1);
 x_axis=floor((position-0.1)/height)+1; 
 y_axis=mod(position,width);
 
 indexnum=0;
 for i=1:6
     for j=1:16
         indexnum=indexnum+1;
          radius=12+(i-1)*8;
          theta=j*2*pi/16;
           xcoe(indexnum)=x_final+radius*cos(theta);
           ycoe(indexnum)=y_final+radius*sin(theta);
       end
 end
 negindex=96;
 neg=zeros(96,400);
   for i=1:negindex
       for j=1:100
           dx=(xcoe(i)-x_axis(j)-dis_vec_total(j,1));  dy=(ycoe(i)-y_axis(j)-dis_vec_total(j,2));
           neg(i, (j-1)*4+1:j*4)=[ min(100,dx^2),  dx , min(100,dy^2), dy  ];
      end
   end
   %% extracted training samples
  pos=zeros(1,400);
  for j=1:100
      dx=x_final-x_axis(j)-dis_vec_total(j,1); dy=y_final-y_axis(j)-dis_vec_total(j,2);
      if dx^2>50||dy^2>50
          pos(1, (j-1)*4+1:j*4)=[0 0 0 0];
          neg(:,(j-1)*4+1:j*4)=0;
      else
          pos(1, (j-1)*4+1:j*4)=[ dx^2, dx  , dy^2 , dy  ];
      end
      
      
  end
  pos=repmat(pos,[5 1]);
  neg=neg*1; pos=pos*1;
 
 


 
 
 

