%%%% AURO - Projet bloc 1 - TP SLAM     %%%%
%%%% Implementation du filtre de Kalman %%%%
%%%% Carrierou Amélie                   %%%%

clear all
close all

%Lancement de la fonction "simulationDonnees pour récupérer les
%observations et les données : N,mX0,PX0,Qw,Rv
[N,T,Z,F,H,mX0,PX0,Qw,Rv,X] = simulationDonnees(0);

%Creation des variables 
H1 = H(1:2,:); % Matrice d'observation pour le cas ou seul A1 est vu 
H2 = H(3:4,:); % Matrice d'observation pour le cas ou seul A2 est vu 
%Creation des variables pour l'etat predit et la matrice de confiance en
%cette prediction
Xpred = cell(1,N);
Ppred = cell(1,N);
%Creation des varables pour l'etat estime et la matrice de confiance en 
%cette estimation 
Xest = cell(1,N);
Pest = cell(1,N);

%Trace de la situation initiale en utilisant la fonction ellipse fournie 
subplot(1,2,1)
plot(mX0(1),mX0(2), 'b+')
hold on 
ellipse(mX0(1:2), PX0(1:2,1:2), 'b')
plot(mX0(3), mX0(4), 'r+')
ellipse(mX0(3:4), PX0(3:4,3:4), 'r')
plot(mX0(5), mX0(6), 'm+')
ellipse(mX0(5:6), PX0(5:6,5:6), 'm')
plot(X(1,1), X(2,1), 'k+')
plot(X(3,1), X(4,1), 'k+')
plot(X(5,1), X(6,1), 'k+')
hold off 
grid on 
title("Elipses de confiance à l'état initial") 
legend("Position robot", "Elipse robot", "Position A1", "Elipse A1", "Position A2", "Elipse A2", "Valeurs reelles")
axis([-6 6 -6  6])

%Initialisation du filtre de Kalman, pas de mesure au premier intant
%Pas de prediction a l'état initial
Xpred{1} = [NaN; NaN; NaN; NaN;]; 
Ppred{1} = NaN;
%Estimation initiale = parametres d'initialisation
Xest{1} = mX0;
Pest{1} = PX0;

%Algo sur N itérations 
for k = 2:N
    %Affichage du numero d'iteration
    disp([newline, 'Itération : ',num2str(k)])
    
    %Prediciton
    Xpred{k} = F*Xest{k-1};
    Ppred{k} = F*Pest{k-1}*F.'+Qw;
    Ppred{k} = (Ppred{k}+Ppred{k}.')/2;
    
    %Mise a jour
    if(~(isnan(Z(1,k)) || isnan(Z(3,k))))
        %Traitement dans le cas ou A1 et A2 sont vus 
        disp("Cas A1 et A2")
        Zpred = H*Xpred{k};
        S = Rv+H*Ppred{k}*(H.');
        K = Ppred{k}*(H.')*inv(S);
        Inov(:,k) = Z(:,k) - Zpred;
        Xest{k} = Xpred{k} + K*(Z(:,k)-Zpred);
        Pest{k} = Ppred{k} - K*H*Ppred{k};
        
    elseif(isnan(Z(3,k)) && ~isnan(Z(1,k)))
        %Traitement dans le cas ou seul A1 est vu
        disp("Cas A1")
        Zpred = H1*Xpred{k};
        S = Rv(1:2,1:2)+H1*Ppred{k}*(H1.');
        K = Ppred{k}*(H1.')*inv(S);
        Inov(:,k) = [Z(1:2,k)-Zpred; NaN; NaN];
        Xest{k} = Xpred{k} + K*(Z(1:2,k)-Zpred);
        Pest{k} = Ppred{k} - K*H1*Ppred{k};
        
    elseif(isnan(Z(1,k)) && ~isnan(Z(3,k)))
        %Traitement dans le cas ou seul A2 est vu
        disp("Cas A2")
        Zpred = H2*Xpred{k};
        S = Rv(3:4,3:4)+H2*Ppred{k}*(H2.');
        K = Ppred{k}*(H2.')*inv(S);
        Inov(:,k) = [NaN;NaN;Z(3:4,k)-Zpred];
        Xest{k} = Xpred{k} + K*(Z(3:4,k)-Zpred);
        Pest{k} = Ppred{k} - K*H2*Ppred{k};
        
    else
        %Traitement quand aucun amer n'est vu
        disp("Cas ni A1 ni A2")
        Inov(:,k) = [NaN;NaN;NaN;NaN];
        Xest{k} = Xpred{k};
        Pest{k} = Ppred{k};
    end
    Pest{k} = (Pest{k}+Pest{k}.')/2;
    
    %Trace de l'elipse a l'iteration k
    subplot(1,2,2)
    %Affichage des estimation pour l'iteration k
    plot(Xest{k}(1),Xest{k}(2), 'b+');
    hold on 
    ellipse(Xest{k}(1:2), Pest{k}(1:2,1:2), 'b');
    plot(Xest{k}(3), Xest{k}(4), 'r+');
    ellipse(Xest{k}(3:4), Pest{k}(3:4,3:4), 'r');
    plot(Xest{k}(5), Xest{k}(6), 'm+');
    ellipse(Xest{k}(5:6), Pest{k}(5:6,5:6), 'm');
    %Affichage des valeurs reelles correspondantes
    plot(X(1,k), X(2,k), 'k+')
    plot(X(3,k), X(4,k), 'k+')
    plot(X(5,k), X(6,k), 'k+')
    hold off 
    grid on 
    title(["Elipses de confiance pour l'iteration ", num2str(k)]) 
    legend("Position robot estimee", "Elipse robot estimee", "Position A1 estimee", "Elipse A1 estimee", "Position A2 estimee", "Elipse A2 estimee", "Valeurs reelles")
    axis([-6 6 -6  6])
    
    pause(0.5)
end    
