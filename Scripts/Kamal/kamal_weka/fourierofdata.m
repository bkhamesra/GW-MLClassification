%generate fourier transform
clear; clc;
 a=csvread('weka_new_notime_nozeros.csv');
 nrows=rows(a);
 ncolumns=columns(a);
 labels=a(:,ncolumns);
 a(:,ncolumns)=[];
 a(1,:)=[];
 
 nrows=rows(a);
 ncolumns=columns(a);
 hcrossindex=ncolumns/2+0;
 
 ncolumns=ncolumns/2;
for n=1:ncolumns
for j=1:nrows
k=n+hcrossindex;
h(j,n)=a(j,n)+i*a(j,k);
end
end
h=h';
%csvwrite('time_cplx_data.csv',h)
htilde=fft(h);
htilde=htilde';
%htilde=[htilde;labels];
htildeReal=real(htilde);
htildeImg=imag(htilde);
htildeseparated=[htildeReal,htildeImg];
aaaa=htildeseparated;j
nrows=rows(htildeseparated);
for j=1:nrows
htildeseparated(j,:)=htildeseparated(j,:)/norm(htildeseparated(j,:));
end

csvwrite('normalized_fourier_frequency_cplx_data.csv',htildeseparated)
