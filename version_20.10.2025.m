%% Segmentazione dei vasi retinici

clc; clear all; close all;
% --- Definizioni dei percorsi ---
basePath = pwd;
inputDir = fullfile(basePath, 'images');
outputDir = fullfile(basePath, 'results_final');
frangiDir = fullfile(basePath, 'frangi_filter_version2a');
addpath(frangiDir);
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
fileList = dir(fullfile(inputDir, '*.tif'));
if isempty(fileList), fileList = dir(fullfile(inputDir, '*.jpg')); end
if isempty(fileList), fileList = dir(fullfile(inputDir, '*.png')); end
disp(['Trovate ', num2str(length(fileList)), ' immagini da elaborare.']);

% --- Parametri Frangi ---
options.FrangiScaleRatio = 0.5;
options.FrangiBetaOne    = 0.5;
options.FrangiBetaTwo    = 8;
options.BlackWhite       = true;

% *************************************************************************
% PARAMETRI TUNING DEFINITIVO (Sensibilità Massima)
% *************************************************************************
% Pesi di Fusione (Bilanciati per connettività e capillari)
w_bh_new = 0.25; w_sato_new = 0.20; 
w_fr_new = max(0, 1 - (w_bh_new + w_sato_new)); % Risulta 0.55

soft_gating_min = 0.35;    % Gating di Coerenza severo
solidity_thr = 0.99;       % Attribute Filtering molto permissivo (anti-frammentazione)

% PARAMETRI ISTERESI CRITICI PER IL RECUPERO
thr_low_prctile = 86;      % Soglia Bassa abbassata per i vasi deboli
hyst_iterations = 6;       % Più iterazioni per colmare i gap

% DISABILITAZIONE FASI RUMOROSE
enable_rethick = false;    
enable_local_thresh = false; 

for i = 1:length(fileList)
    fileName = fileList(i).name;
    fullPath = fullfile(inputDir, fileName);
    disp(['Elaborazione: ', fileName, ' (', num2str(i), '/', num2str(length(fileList)), ')']);
    
    %% 1) Lettura + FOV
    I = imread(fullPath);
    if size(I,3)==3, I_gray = rgb2gray(I); else, I_gray = I; end
    I_double = im2double(I_gray);
    avg_intensity = mean(I_double(:));
    simple_mask = I_double > avg_intensity * 0.3;
    I_eq_soft = histeq(I_double, 256);
    I_clahe   = adapthisteq(I_double,'ClipLimit',0.02,'NumTiles',[8 8],'Distribution','uniform');
    I_eq_soft(~simple_mask) = 0;  I_clahe(~simple_mask) = 0;
    
    FOV = I_double > graythresh(I_double)*0.2;
    FOV = imfill(FOV,'holes');
    FOV = bwareafilt(FOV,1);
    FOV_core = imerode(FOV, strel('disk', 8));
    border   = FOV & ~FOV_core;
    FOV_in   = FOV_core;
    
    %% 1b) Immagine dedicata all'enhancement (CLAHE più localizzato)
    if size(I,3)==3, Ig = I(:,:,2); else, Ig = I; end
    Ig = im2double(Ig);
    R  = round(min(size(Ig))/30);
    bg = imopen(Ig, strel('disk', max(R,12)));
    I_enh = imadjust(Ig - bg);
    I_enh = imdiffusefilt(I_enh,'GradientThreshold',8,'NumberOfIterations',4);
    I_enh = adapthisteq(I_enh,'ClipLimit',0.01,'NumTiles',[16 16]);
    I_enh(~FOV_in) = 0;
    
    %% 2) Enhancement: Frangi + bottom-hat (NMS) + Sato
    opt = options;  V_fr = zeros(size(I_enh));
    scaleBands = {[0.8 1.8], [1.5 3.0], [2.5 5.0]};
    for k = 1:numel(scaleBands)
        opt.FrangiScaleRange = scaleBands{k};
        V_fr = max(V_fr, FrangiFilter2D(I_enh, opt));
    end
    V_fr = rescale(V_fr);
    V_bh = directional_bothat_nms(I_enh, FOV_in, [7 11 15 19], 0:15:165, 11);
    [Gx,Gy] = imgradientxy(I_enh,'sobel');
    J11 = imgaussfilt(Gx.^2,1); J22 = imgaussfilt(Gy.^2,1); J12 = imgaussfilt(Gx.*Gy,1);
    coh = (sqrt((J11-J22).^2 + 4*J12.^2)) ./ (J11+J22+eps);
    V_bh = V_bh .* (0.3 + 0.7*mat2gray(coh));
    V_bh = rescale(V_bh);
    V_sato = sato_vesselness2d_multiscale(I_enh, [1.0 2.0 3.5], true);
    V_sato = rescale(V_sato);
    
    % Fusione con i NUOVI PESI
    fused = w_fr_new*V_fr + w_bh_new*V_bh + w_sato_new*V_sato;
    vals_all = fused(FOV_in);
    
    % Normalizzazione robusta PIÙ INCLUSIVA
    lo = prctile(vals_all,5); 
    hi = prctile(vals_all,99.9); 
    vessel_fused = mat2gray(fused,[lo hi]).^1.0; 
    vessel_fused(~FOV_in) = 0; vessel_fused(border) = 0;
    
    %% 3) Anti-neve + gating di coerenza (Più Severo)
    E = entropyfilt(I_enh, true(9));
    S = stdfilt(I_enh, true(5));
    tex = (E > prctile(E(FOV_in),20)) | (S > prctile(S(FOV_in),20));
    vessel_fused(~tex) = 0;
    vessel_fused = vessel_fused .* (soft_gating_min + (1-soft_gating_min)*mat2gray(coh));
    vessel_fused = imtophat(vessel_fused, strel('disk',2));
    
    %% 4) Soglia: Isteresi Pura POTENZIATA (RECUPERO)
    vals = vessel_fused(FOV_in);
    thr_high = prctile(vals, 97);
    thr_low  = prctile(vals, thr_low_prctile); % Usa 86° percentile
    BW_high  = (vessel_fused > thr_high) & FOV_in;
    BW_low   = (vessel_fused > thr_low)  & FOV_in;
    
    % Isteresi direzionale con più iterazioni e strel più lungo
    BW_hyst_dir = hysteresis_directional_potenziata(BW_high, BW_low, 0:15:165, hyst_iterations);
    
    if enable_local_thresh
        % Questa sezione rimane disabilitata per prevenire il rumore
        uncertain = (vessel_fused > thr_low*0.9) & (vessel_fused < thr_high) & FOV_in;
        Tloc = adaptthresh(vessel_fused, 0.52, 'NeighborhoodSize', [35 35], 'Statistic','mean');
        BW_adapt_all = imbinarize(vessel_fused, Tloc);
        BW_adapt = BW_adapt_all & uncertain;
        BW_refined = BW_hyst_dir | BW_adapt;
    else
        BW_refined = BW_hyst_dir;
    end
    
    %% 5) Attribute filtering (anti-frammentazione) + closing direzionale
    CC = bwconncomp(BW_refined);
    Sprops = regionprops(CC,'Eccentricity','Solidity','Area');
    ecc = [Sprops.Eccentricity]; sol = [Sprops.Solidity]; area = [Sprops.Area];
    
    % Attribute Filtering AMMORBIDITO (anti-frammentazione)
    keep = (ecc > 0.35 | area > 10) & (sol < solidity_thr); 
    
    BW_refined = ismember(labelmatrix(CC), find(keep));
    BW_refined = bwareaopen(BW_refined, 6);
    for th = [0 45 90 135]
        BW_refined = imclose(BW_refined, strel('line',3,th));
    end
    
    %% 6) Re-thickening (DISABILITATO)
    if enable_rethick
        Sk = bwskel(BW_refined,'MinBranchLength',5);
        core = imdilate(Sk, strel('disk',1));
        rethick = core & (vessel_fused > thr_low*0.9);
        final_mask = (BW_refined | rethick) & FOV;
    else
        final_mask = BW_refined & FOV;
    end
    
    %% 7) Recupero (cauto) presso il bordo
    edge_zone = imdilate(~FOV_in, strel('disk',5));
    near_edge = imdilate(edge_zone, strel('disk',10)) & ~edge_zone;
    faint_candidates = (vessel_fused > thr_low*0.95) & near_edge;
    final_mask = (final_mask | faint_candidates) & FOV;
    
    %% 8) Salvataggio e figure
    [~, name, ~] = fileparts(fileName);
    imwrite(final_mask, fullfile(outputDir,[name '_segmented_final_max_sens.png']));
    figure('Position',[50 50 1500 600]);
    subplot(2,4,1); imshow(I);          title('Originale');
    subplot(2,4,2); imshow(I_gray);     title('Scala di Grigi');
    subplot(2,4,3); imshow(I_eq_soft);  title('Equalizzato Soft');
    subplot(2,4,4); imshow(I_clahe);    title('CLAHE Soft');
    subplot(2,4,5); imshow(vessel_fused,[]); title('Mappa (Fusione Potenziata)');
    subplot(2,4,6); imshow(BW_hyst_dir);     title('Isteresi Direzionale (Potenziata)');
    subplot(2,4,7); imshow(BW_refined);      title('Dopo Filtering Ammorbidito');
    subplot(2,4,8); imshow(final_mask);      title('Segmentazione Finale');
    sgtitle(['Segmentazione CONNESSIONE MAX: ', fileName], 'Interpreter','none');
end
rmpath(frangiDir);
disp('--- Elaborazione completata! ---');

% ================== FUNZIONI DI SUPPORTO ==================
function V = directional_bothat_nms(I, mask, lens, thetas, nms_len)
% Bottom-hat orientato multi-scala con Non-Maximum Suppression (NMS)
    V_or = zeros([size(I), numel(thetas)]);
    for ti = 1:numel(thetas)
        th = thetas(ti); Rtheta = zeros(size(I));
        for L = lens
            se = strel('line', L, th); Rtheta = max(Rtheta, imbothat(I, se));
        end
        seN  = strel('line', max(nms_len,3), th); local_max = imdilate(Rtheta, seN);
        Rtheta_nms = Rtheta .* (Rtheta >= local_max - 1e-6); V_or(:,:,ti) = Rtheta_nms;
    end
    V = max(V_or, [], 3); V(~mask) = 0;
end
function BW = hysteresis_directional_potenziata(BW_high, BW_low, thetas, n_iter)
% Isteresi con crescita limitata e direzionale (strel più lungo, più iterazioni)
    BW = BW_high;
    mask = BW_low;
    for it = 1:n_iter
        grown = false(size(BW));
        for th = thetas
            grown = grown | (imdilate(BW, strel('line',4,th)) & mask); % STREL DA 3 A 4
        end
        if isequal(grown, BW), break; end
        BW = grown;
    end
end
function V = sato_vesselness2d_multiscale(I, sigmas, darkRidges)
% Vesselness tipo Sato (Hessian-based) multiscala
    I = im2double(I); V = zeros(size(I)); beta = 0.5;  c = 1.0;
    for s = sigmas
        [Dxx, Dxy, Dyy] = hessian2D(I, s); [L1, L2] = eig2image(Dxx, Dxy, Dyy);
        if darkRidges, L2(L2 > 0) = 0; else, L2(L2 < 0) = 0; end
        RB = abs(L1) ./ (abs(L2) + eps); S  = sqrt(L1.^2 + L2.^2);
        V_s = exp(-(RB.^2)/(2*beta^2)) .* (1 - exp(-(S.^2)/(2*c^2))); V = max(V, V_s);
    end; V = rescale(V);
end
function [Dxx, Dxy, Dyy] = hessian2D(I, sigma)
% Hessian 2D via derivate di Gauss
    halfsize = ceil(3*sigma); [x,y] = meshgrid(-halfsize:halfsize, -halfsize:halfsize);
    gauss = exp(-(x.^2 + y.^2)/(2*sigma^2)) / (2*pi*sigma^2);
    Dxx_k = (x.^2/sigma^4 - 1/sigma^2) .* gauss;
    Dyy_k = (y.^2/sigma^4 - 1/sigma^2) .* gauss;
    Dxy_k = (x.*y/sigma^4) .* gauss;
    Dxx = imfilter(I, Dxx_k, 'replicate', 'conv');
    Dyy = imfilter(I, Dyy_k, 'replicate', 'conv');
    Dxy = imfilter(I, Dxy_k, 'replicate', 'conv');
end
function [Lambda1, Lambda2] = eig2image(Dxx, Dxy, Dyy)
% Autovalori ordinati per modulo: |Lambda1| >= |Lambda2|
    tmp = sqrt( (Dxx - Dyy).^2 + 4*Dxy.^2 ); v2x = 2*Dxy; v2y = Dyy - Dxx + tmp;
    mag = sqrt(v2x.^2 + v2y.^2) + eps; v2x = v2x ./ mag; v2y = v2y ./ mag;
    v1x = -v2y; v1y = v2x;
    Lambda1 = v1x.^2.*Dxx + 2*v1x.*v1y.*Dxy + v1y.^2.*Dyy;
    Lambda2 = v2x.^2.*Dxx + 2*v2x.*v2y.*Dxy + v2y.^2.*Dyy;
    swap = abs(Lambda1) < abs(Lambda2); L1 = Lambda1; L2 = Lambda2;
    Lambda1(swap) = L2(swap); Lambda2(swap) = L1(swap);
end