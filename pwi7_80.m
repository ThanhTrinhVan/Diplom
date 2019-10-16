%��������� ���������� 24-� �������������� �������� ���������
%������� ���������� ����������� (��� ���������� R, G, B, RG,RB, GB ��������� 
%� ��� ���������� ����������� ����������)
%���������� glcm � ���������� ��� �������� ������������
%R,G,B (inter_channel)� ��� ���������� ������������ RG, RB, GB, (intra_channel)
%���������� ����� ��������� ������ R, B, G

clc;
clear;
%�������������� ����� ��������������� ����������� ������������ 
%100�300 �������� 
%1-15.jpg 


he = imread('1.jpg');%�������� (healthy) ��������

%��������� r,g,b ���������
r=he(:,:,1);
g=he(:,:,2);
b=he(:,:,3);

%���������� inter_channel_matrix
rgb_image=im2double(he);%�������������� ��������� ����������� � ������ double
%figure, imshow(he);
fR = rgb_image (:, :, 1);
fG = rgb_image (:, :, 2);
fB = rgb_image (:, :, 3);

%���������� intra_channel_matrix ����� ��������� ������
fRG=fR-fG;
fRB=fR-fB;
fGB=fG-fB;



%���������� glcm � ���������� ��� �������� ������������
%R,G,B (inter_channel)
glcm = graycomatrix(fR, 'Offset',[2 0]);%���������� ������� glcm
stats_R = graycoprops(glcm)%���������� �������������� ������������� glcm
glcm = graycomatrix(fG, 'Offset',[2 0]);%���������� ������� glcm
stats_G = graycoprops(glcm)%���������� �������������� ������������� glcm
glcm = graycomatrix(fB, 'Offset',[2 0]);%���������� ������� glcm
stats_B = graycoprops(glcm)%���������� �������������� ������������� glcm

%���������� glcm � ���������� ��� ���������� ������������ RG, RB, GB, (intra_channel)
%���������� ����� ��������� ������ fR, fB, fG
glcm = graycomatrix(fRG, 'Offset',[2 0]);%���������� ������� glcm
stats_RG = graycoprops(glcm)%���������� �������������� ������������� glcm
glcm = graycomatrix(fRB, 'Offset',[2 0]);%���������� ������� glcm
stats_RB = graycoprops(glcm)%���������� �������������� ������������� glcm
glcm = graycomatrix(fGB, 'Offset',[2 0]);%���������� ������� glcm
stats_GB = graycoprops(glcm)%���������� �������������� ������������� glcm

%pause;
close all;
clear;




