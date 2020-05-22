%clear all;

% figure,imshow(DJI_0178);
% hold on;
% plot(3150,1958, 'ro', 'MarkerSize', 3);
% imtool(DJI_0178);

%%
% register=[iamgeID, p2_point[x,y],image_point[x,y],rowSpace]
%load('excelFile.mat')
rotated_img=load('Img_folder_address.mat');
rotated_img=rotated_img.img_path;
excelFile_address=load('label_file_address.mat');
excelFile_address=excelFile_address.label_path;
excelFile=readtable(excelFile_address);
excelFile=table2array(excelFile);
rowSpace_address=load('rowSpace_file_address.mat');
rowSpace_address=rowSpace_address.rowSpace_path;
register=readtable(rowSpace_address,'ReadVariableNames',false);
register=table2array(register);
output_path='.\meter\';

i=1;
flag=0;
Names=[];
label=[];
area_list=[];
crop_length_1=0.7;
crop_length_2=1.3;




for registerID=1:size(register,1)
    p2_point=register(registerID,2:3);
    image_point=register(registerID,4:5);
    
    rowSpace=register(registerID,6);
    id=register(registerID,1);
    %excelFile=[piont image	roi	label	cx	bx	cy  by	bw	bh]

    
    p1=imread(strcat(rotated_img,'DJI_0',int2str(id),'.JPG'));
    point=excelFile(i,1);
    dx=image_point(1)-p2_point(1);
    dy=image_point(2)-p2_point(2);
    excelFile(:,5:6)=excelFile(:,5:6)+dx;
    excelFile(:,7:8)=excelFile(:,7:8)+dy;

    
    while 1
        if i>size(excelFile,1)
            break;
        end
        
        if excelFile(i,2)~=id
            break;
        end

        top=excelFile(i,8);
        l=1;
        imageSize=0;
        seedlingNum=excelFile(i,4);
        area=excelFile(i,11);
        flag=0;
        if i+l>size(excelFile,1)
            break;
        end
        if excelFile(i+l,8)<excelFile(i,8)   %line change
               i=i+l+1;
               continue;
        end
        while (excelFile(i+l,8)+excelFile(i+l,10)-top)<(rowSpace*crop_length_1)
          
            if excelFile(i+l+1,8)>excelFile(i+l,8)
               seedlingNum=seedlingNum+excelFile(i+l,4);
               area=area+excelFile(i+l,11);
               l=l+1;
            else
               flag=1;
               break;
            end
        end
        
        seedlingNum=seedlingNum+excelFile(i+l,4);
        area=area+excelFile(i+l,11);
        imageSize=(excelFile(i+l,8)+excelFile(i+l,10)-top);
        

        if imageSize<200  %line change
               i=i+l+1;
               continue;
        end
        
        
        r=randi([max(floor(excelFile(i,6)-imageSize/2-1),1),max(floor(excelFile(i,6)-imageSize/2+1),1)]);

        if r+imageSize>size(p1,2)
                r=size(p1,2)-imageSize-1;
        end
            
        p=1;
       
        temp=p1(max(floor(excelFile(i,8)),1):min(floor(excelFile(i+l,8)+excelFile(i+l,10)),size(p1,1)),r:r+imageSize,:);
        fName=strcat(num2str(point),'00',num2str(id),'00',num2str(excelFile(i,3)),'00',num2str(excelFile(i+l,3)),'00',num2str(imageSize),'00',num2str(p),'.tiff');

        imwrite(temp,strcat(output_path,fName),'tiff','Compression', 'none');
        Names=[Names;string(fName)];
        label=[label; seedlingNum];
        area_list=[area_list;area];
        
        if flag==1
            i=i+l+1;
            continue;
        end
        
        l=l+1;
        
        if excelFile(i+l,8)<excelFile(i,8)   %line change
               i=i+l+1;
               continue;
        end
        
        while (excelFile(i+l,8)+excelFile(i+l,10)-top)<(rowSpace*crop_length_2)
            
            p=p+1;
            imageSize=(excelFile(i+l,8)+excelFile(i+l,10)-top);
            seedlingNum=seedlingNum+excelFile(i+l,4);
            area=area+excelFile(i+l,11);
            
            
            
            r=randi([max(floor(excelFile(i,6)-imageSize/2-1),1),max(floor(excelFile(i,6)-imageSize/2+1),1)]);

            
            if r+imageSize>size(p1,2)
                r=size(p1,2)-imageSize-1;
            end
       
            temp=p1(max(floor(excelFile(i,8)),1):min(floor(excelFile(i+l,8)+excelFile(i+l,10)),size(p1,1)),r:r+imageSize,:);
            fName=strcat(num2str(point),'00',num2str(id),'00',num2str(excelFile(i,3)),'00',num2str(excelFile(i+l,3)),'00',num2str(imageSize),'00',num2str(p),'.tiff');

            imwrite(temp,strcat(output_path,fName),'tiff','Compression', 'none');
            Names=[Names;string(fName)];
            label=[label; seedlingNum];
            area_list=[area_list;area];
            
            if (i+l+1<size(excelFile,1))&(excelFile(i+l+1,8)>excelFile(i+l,8))
               l=l+1;
            else
               flag=1;
               break;
            end
            
        end

        if flag==1
            i=i+l+1;
            continue;
        end
        i=i+1;
    end
    
    excelFile(:,5:6)=excelFile(:,5:6)-dx;
    excelFile(:,7:8)=excelFile(:,7:8)-dy;

end

result=[Names,label,area_list];
T = array2table(result,'VariableNames',{'name','labels','area'});
writetable(T,'LabelsForMeter.csv');