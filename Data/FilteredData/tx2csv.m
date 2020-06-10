clear all;clc;
s=dir('*.txt');


%file1 = '/home/pingo/GW_Waveforms/Metadata/myfile.csv';
%M     = csvread(file1,1,0);
metadata=dir('/home/pingo/GW_Waveforms/Metadata/*.csv');
%m=length(metadata);

n=length(s); %total number of rows in s, i.e., total number of files in s
M = cell(n, 1);


%for k=1:length(s)
%q=0;
%for p=1:n
%q=max(q,length(s(p))
for k=1:n
a=dlmread(s(k).name);
a(:,5)=[];
a(:,4)=[];
size(a); 
%a=a';
%M{k}=a;
%weka(k,:) = M{k}(:)';
metafilename=["/home/pingo/GW_Waveforms/Metadata/Metadata_" s(k).name(1:end-3) "csv"];
[b] = textread (metafilename,"%s");
%%%%%%%%%%%%%%%%%%%%%%%%%metafilename
%%%%%%%%%%%%%%%%%%%%%%%%%metadatacurrent=b(21,1){1};
row = a(:)';
A=fileread(metafilename);
B=strfind(A, "spin-type,");
spintype=A(1,B+10);


s(k).name

if spintype=="P" 
weka(k,:)=[row, 9999];


elseif spintype=="N" 
weka(k,:)=[row, 2222];


elseif spintype=="A" 
weka(k,:)=[row, 1111];

else
s(k).name  %print filenames that are not satisfying any conditions
end
%[v,w]=size(weka);
%weka(k,:)=[colon(1,v);weka(k,:)]


%%%%if strcmp(metadatacurrent,"spin-type,Precessing")==1 || strcmp(b(21,1){1},"spin-type,Precessing")==1
%%%%weka(k,:)=[row, 9876];


%%%%%elseif strcmp(metadatacurrent,"spin-type,AlignedSpins")==1
%%%%weka(k,:)=[row, 5432];


%%%%elseif strcmp(metadatacurrent,"spin-type,NonSpinning")==1
%%%%weka(k,:)=[row, 101010];

%%%%else
%%%%s(k).name
%%%%end

end
csvwrite('weka_new_edited.csv',weka)

%M{1}
%M{2}
%M{3}
%for k=1:length(s)
%a=dlmread(s(k).name);
%M{k}=a.data;
%end

%plot(M{3}(:,2),M{3}(:,3))
%weka(1,:) = M{1}(:)';
%csvwrite('weka.csv',weka)