%path='E:\spyder\row_4\';

path=load('Img_folder_address.mat');
output=load('output_file_address.mat');
path=path.img_path;
output=output.output_path;
%points=[4;8;12;16;20;24;28];
points=[4];
% points=[3;7;11;15;19;23;27];
% points=[2;6;10;14;18;22;26];
segment_start_piont=[];
row_space=[];
excel_data=[];
img_name=[];


for kk=1:size(points,1)
    myFolder = strcat(path,'p',num2str(points(kk)),'\');
    filePattern = fullfile(myFolder);
    jpegFiles = dir(filePattern);
    
    for kkk=3:length(jpegFiles)
        p1=imread(strcat(jpegFiles(kkk).folder,'\',jpegFiles(kkk).name));
    
        p3=p1;

        for i=1:size(p3,1)    % removed the effect of PVC
            for j=1:size(p3,2)
                if (p3(i,j,1)>200)||(p3(i,j,3)>200)
                  p3(i,j,:)=0;
                end
            end
        end
        % imtool(p3);

        p3p=decorrstretch(p3);
        p4p=p3p(:,:,2);
        p4p(p4p<180)=0;
        % imtool(p4p);

        L=p4p;
        L=bwareaopen(L, 30);
        % hough line detection
        [H,T,R] = hough(L,'RhoResolution',1,'Theta',-90:0.5:89);

        P  = houghpeaks(H,15,'threshold',ceil(0.01*max(H(:))));
        x = T(P(:,2)); y = R(P(:,1));

        M=mode(x); %the cotton row angle
        L = imrotate(L,M);
        
        img_rotate = imrotate(p1,M);
        imwrite(img_rotate,strcat(output,jpegFiles(kkk).name));
        
        gcf=figure, imshow(img_rotate);
        [x1,y1]=ginput(1);
        [x2,y2]=ginput(1);
        segment_start_piont=[segment_start_piont;x1,y1];
        
        close(gcf);
        segment=img_rotate(y1:y2,x1:x2,:);
        segment_L=L(y1:y2,x1:x2,:);
        
        B=[0 1 0
           1 1 1
           0 1 0];
        for i=1:3
            segment_L=imdilate(segment_L,B);
        end
        
        stats = regionprops(segment_L,'BoundingBox','centroid','Area');
        
        %sort the stats
        centroids=cat(1, stats.Centroid);
        ind=find(centroids(:,1)>size(segment,2)/2);
        stats1=stats(1:ind(1)-1);
        stats2=stats(ind(1):ind(end));

        centroids1=cat(1, stats1.Centroid);
        [x,idx1]=sort(centroids1(:,2));
        stats1=stats1(idx1);

        centroids2=cat(1, stats2.Centroid);
        [x,idx2]=sort(centroids2(:,2));
        stats2=stats2(idx2);

        stats=[stats1;stats2];

        centroids=cat(1, stats.Centroid);
        boundingBox=cat(1, stats.BoundingBox);
        area=cat(1, stats.Area);
        temp=zeros(size(centroids,1),11);
        temp(:,1)=points(kk);
        a=strsplit(jpegFiles(kkk).name,{'DJI_','.JPG'});
        temp(:,2)=str2num(a{2});
        temp(:,3)=1:size(stats,1);
        temp(:,5)=centroids(:,1);
        temp(:,7)=centroids(:,2);
        temp(:,6)=boundingBox(:,1);
        temp(:,8:10)=boundingBox(:,2:4);
        temp(:,11)=area;
        excel_data=[excel_data;temp];
        
        gcf=figure('visible','off'), imshow(segment);%impixelinfo
        for i=1:size(centroids,1)
            rectangle('Position',boundingBox(i,:),'EdgeColor','b','LineWidth',2); 
            H=text(floor(centroids(i,1)), floor(centroids(i,2)),num2str(i), 'color','r');
        end
        saveas(gcf,strcat(path,jpegFiles(kkk).name,'.png'));
        
        [H,T,R] = hough(L,'RhoResolution',1,'Theta',-7:7:7);

        P  = houghpeaks(H,15,'threshold',ceil(0.01*max(H(:))));
        x = T(P(:,2)); y = R(P(:,1));
        
        for tt=size(x,2):-1:1
            if x(tt)~=0
                P(tt,:)=[];
            end
        end

        x = T(P(:,2)); y = R(P(:,1));
%         lines = houghlines(L,T,R,P,'FillGap',8,'MinLength',7);
%         figure, imshow(L),impixelinfo; hold on
%         for k = 1:length(lines)
%            xy = [lines(k).point1; lines(k).point2];
%            plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
% 
%            % Plot beginnings and ends of lines
%            plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%            plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
% 
%         end
        y2=sort(y');
        y3=y2(2:end);
        this_row=sort(y3-y2(1:end-1));
        row_space=[row_space;this_row(end-1)];
        img_name=[img_name;str2num(a{2})];
    end
end

%save 'excel_data.mat' excel_data;
%save 'row_space.mat' row_space;
%save 'segment_start_piont.mat' segment_start_piont;

T = array2table(excel_data,'VariableNames',{'GRPs','img_name','cluster_id','label','cx','bx','cy','by','bw','bh','area'});
writetable(T,'excel_data.csv');

S=zeros(size(row_space,1),6);
S(:,6)=row_space;
S(:,4:5)=segment_start_piont;
S(:,1)=img_name;
%writematrix(S,'row_space.csv');

S1=array2table(S);
writetable(S1,'row_space.csv','WriteVariableNames',false);
