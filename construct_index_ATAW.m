function [ATAW_positive_index, ATAW_negative_index]=construct_index_ATAW(Ap)

          tmp1=unique(Ap(:,1)); tmp2=unique(Ap(:,2));
          for i=1:length(tmp1)
                ATAW_positive_index(i,1)=tmp1(i);
          end
          for i=1:length(tmp1)
              index=find( Ap(:,1)==ATAW_positive_index(i,1)  );
              ATAW_positive_index(i,2:2+length(index)-1)=index';
          end
          for i=1:length(tmp2)
                ATAW_negative_index(i,1)=tmp2(i);
          end
           for i=1:length(tmp2)
              index=find( Ap(:,2)==ATAW_negative_index(i,1)  );
              ATAW_negative_index(i,2:2+length(index)-1)=index';
          end