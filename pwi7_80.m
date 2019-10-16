%Программа вычисления 24-х статистических ключевых признаков
%цветных текстурных изображений (без фильтрации R, G, B, RG,RB, GB компонент 
%и без адаптивной эквализации гистограмм)
%Вычисление glcm и статистики для цветовых составляющих
%R,G,B (inter_channel)и для разностных составляющих RG, RB, GB, (intra_channel)
%полученных путем вычитания матриц R, B, G

clc;
clear;
%Обрабатываться будут нормализованные изображения размерностью 
%100х300 пикселей 
%1-15.jpg 


he = imread('1.jpg');%здоровое (healthy) растение

%выделение r,g,b компонент
r=he(:,:,1);
g=he(:,:,2);
b=he(:,:,3);

%вычисление inter_channel_matrix
rgb_image=im2double(he);%преобразование элементов изображения в формат double
%figure, imshow(he);
fR = rgb_image (:, :, 1);
fG = rgb_image (:, :, 2);
fB = rgb_image (:, :, 3);

%вычисление intra_channel_matrix путем вычитания матриц
fRG=fR-fG;
fRB=fR-fB;
fGB=fG-fB;



%Вычисление glcm и статистики для цветовых составляющих
%R,G,B (inter_channel)
glcm = graycomatrix(fR, 'Offset',[2 0]);%вычисление матрицы glcm
stats_R = graycoprops(glcm)%вычисление статистических характеристик glcm
glcm = graycomatrix(fG, 'Offset',[2 0]);%вычисление матрицы glcm
stats_G = graycoprops(glcm)%вычисление статистических характеристик glcm
glcm = graycomatrix(fB, 'Offset',[2 0]);%вычисление матрицы glcm
stats_B = graycoprops(glcm)%вычисление статистических характеристик glcm

%Вычисление glcm и статистики для разностных составляющих RG, RB, GB, (intra_channel)
%полученных путем вычитания матриц fR, fB, fG
glcm = graycomatrix(fRG, 'Offset',[2 0]);%вычисление матрицы glcm
stats_RG = graycoprops(glcm)%вычисление статистических характеристик glcm
glcm = graycomatrix(fRB, 'Offset',[2 0]);%вычисление матрицы glcm
stats_RB = graycoprops(glcm)%вычисление статистических характеристик glcm
glcm = graycomatrix(fGB, 'Offset',[2 0]);%вычисление матрицы glcm
stats_GB = graycoprops(glcm)%вычисление статистических характеристик glcm

%pause;
close all;
clear;




