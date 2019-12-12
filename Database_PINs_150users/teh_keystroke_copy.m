%%% Script processing Keytsroke Data with CNN / RNN


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% listing = dir('D:\University\Biometrics\Programmi\Keystroke\DeepKey\Short_DG150\');
% 
% for k = 3:length(listing)
% 
%     FileName = ['D:\University\Biometrics\Programmi\Keystroke\DeepKey\Short_DG150\'  listing(k).name];
% 
%     eval(['Database_ShortPINs.User' int2str(k-2) ' = readmatrix(FileName);'])
% 
% end
% 
% 
% listing = dir('D:\University\Biometrics\Programmi\Keystroke\DeepKey\Long_DG150\');
% 
% for k = 3:length(listing)
% 
%     FileName = ['D:\University\Biometrics\Programmi\Keystroke\DeepKey\Long_DG150\'  listing(k).name];
% 
%     eval(['Database_LongPINs.User' int2str(k-2) ' = readmatrix(FileName);'])
% 
% end
% 
% 
% save('Database_PINs_150users.mat','Database_ShortPINs','Database_LongPINs')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load('Database_PINs_150users.mat')


%%%


Dataset = Database_LongPINs;


%%%


Train_Data_CNN_KP = []; Train_Data_CNN_KR = []; Train_Data_CNN_KA = []; Train_Data_CNN_KPR = [];

Test_Data_CNN_KP = []; Test_Data_CNN_KR = []; Test_Data_CNN_KA = []; Test_Data_CNN_KPR = [];

Train_Data_RNN_KP = []; Train_Data_RNN_KR = []; Train_Data_RNN_KA = []; Train_Data_RNN_KPR = [];

Test_Data_RNN_KP = []; Test_Data_RNN_KR = []; Test_Data_RNN_KA = []; Test_Data_RNN_KPR = [];

 

Train_Data_CNN = []; Test_Data_CNN = [];

Train_Data_RNN = []; Test_Data_RNN = [];

Train_Label_CNN = []; Test_Label_CNN = [];

Train_Label_RNN = []; Test_Label_RNN = [];


Train_set = [1:3 6:10]; Test_set = setdiff(1:10,Train_set); % Best results for testoing with acquisitions 5, 4, 7, 6, 8


Ns = 10;

Nk = size(Dataset.User1,1)/Ns;

Nv = size(Dataset.User1,2);

Nt = length(Train_set);

disp(Nk);
disp(Nv);
disp(Nt);


for u = 1:150

    % eval(['data = USER' int2str(u) '.Variables;'])

    eval(['data = Dataset.User' int2str(u) ';'])

    data = reshape(data',Nv,Nk,Ns);
    
    disp(data);
    
%     disp (data');

    % data = permute(data,[2 1 3]);

    % data = data(1:Nk-1,:,:); % remove the information regarding the press of the ENTER key

    KP(1,:,1,:) = single(round(data(3,:,:)/1000)/1000000);

    KR(1,:,1,:) = single(round(data(5,:,:)/1000)/1000000);

    KA(1,:,1,:) = single(data(6,:,:));

    % KPR(1,1:2:33,1,:) = KP; KPR(1,2:2:34,1,:) = KR;

   

    KH = KR - KP;

    KPP = KP(1,2:Nk,1,:) - KP(1,1:Nk-1,1,:);

    KPR = KR(1,2:Nk,1,:) - KP(1,1:Nk-1,1,:);

    KRP = KP(1,2:Nk,1,:) - KR(1,1:Nk-1,1,:);

    KRR = KR(1,2:Nk,1,:) - KR(1,1:Nk-1,1,:);   

    

%     KP_P = KP(1,3:Nk,1,:) - KP(1,1:Nk-2,1,:);

%     KP_R = KR(1,3:Nk,1,:) - KP(1,1:Nk-2,1,:);

%     KR_P = KP(1,3:Nk,1,:) - KR(1,1:Nk-2,1,:);

%     KR_R = KR(1,3:Nk,1,:) - KR(1,1:Nk-2,1,:);

%        

%     user_data = cat(1,  KH(1,1:Nk-2,1,:), ...

%                         ...

%                         KPP(1,1:Nk-2,1,:),...

%                         KPR(1,1:Nk-2,1,:),...                   

%                         KRP(1,1:Nk-2,1,:),...

%                         KRR(1,1:Nk-2,1,:),...

%                         KH(1,2:Nk-1,1,:),...

%                         ...

%                         KP_P(1,1:Nk-2,1,:),...

%                         KP_R(1,1:Nk-2,1,:),...                   

%                         KR_P(1,1:Nk-2,1,:),...

%                         KR_R(1,1:Nk-2,1,:),...

%                         KH(1,3:Nk,1,:),...

%                         ...

%                         KA(1,1:Nk-2,1,:),...

%                         KA(1,2:Nk-1,1,:),...

%                         KA(1,3:Nk,1,:));                    

                

    user_data = cat(1,  KH(1,1:Nk-1,1,:), ... 
                        ... 
                        KPP(1,1:Nk-1,1,:),...
                        KPR(1,1:Nk-1,1,:),...                   
                        KRP(1,1:Nk-1,1,:),...
                        KRR(1,1:Nk-1,1,:),...
                        KH(1,2:Nk,1,:),...
                        ...
                        KA(1,1:Nk-1,1,:),...
                        KA(1,2:Nk,1,:));  
                    
   


   user_data = cat(1,KH(1,1:Nk-1,1,:),KPP(1,1:Nk-1,1,:),KPR(1,1:Nk-1,1,:),KRP(1,1:Nk-1,1,:),KRR(1,1:Nk-1,1,:));
   disp(user_data)
   
   writematrix(user_data,'merged.csv')
%    type 'merged.csv'

end

%     Train_Data_CNN_KP  = cat(4,Train_Data_CNN_KP,KP(1,:,1,Train_set));

%     Train_Data_CNN_KR  = cat(4,Train_Data_CNN_KR,KR(1,:,1,Train_set));

%     Train_Data_CNN_KA  = cat(4,Train_Data_CNN_KA,KA(1,:,1,Train_set));

%     Train_Data_CNN_KPR = cat(4,Train_Data_CNN_KPR,KPR(1,:,1,Train_set));

%    

%     Test_Data_CNN_KP  = cat(4,Test_Data_CNN_KP,KP(1,:,1,Test_set));

%     Test_Data_CNN_KR  = cat(4,Test_Data_CNN_KR,KR(1,:,1,Test_set));

%     Test_Data_CNN_KA  = cat(4,Test_Data_CNN_KA,KA(1,:,1,Test_set));

%     Test_Data_CNN_KPR = cat(4,Test_Data_CNN_KPR,KPR(1,:,1,Test_set));  

%    

%     for t = 1:8

%         Train_Data_RNN_KP{end+1}  = KP(1,:,1,Train_set(t));

%         Train_Data_RNN_KR{end+1}  = KR(1,:,1,Train_set(t));

%         Train_Data_RNN_KA{end+1}  = KA(1,:,1,Train_set(t));

%         Train_Data_RNN_KPR{end+1} = KPR(1,:,1,Train_set(t));

%     end

%    

%     for t = 1:2

%         Test_Data_RNN_KP{end+1}  = KP(1,:,1,Test_set(t));

%         Test_Data_RNN_KR{end+1}  = KR(1,:,1,Test_set(t));

%         Test_Data_RNN_KA{end+1}  = KA(1,:,1,Test_set(t));

%         Test_Data_RNN_KPR{end+1} = KPR(1,:,1,Test_set(t));

%     end


%     Train_Data_CNN = cat(4,Train_Data_CNN,user_data(:,:,1,Train_set));
% 
%     Test_Data_CNN = cat(4,Test_Data_CNN,user_data(:,:,1,Test_set));
% 
%    
% 
%     for t = 1:Nt
% 
%         Train_Data_RNN{end+1}  = user_data(:,:,1,Train_set(t));
% 
%         Train_Label_RNN{end+1} = categorical(u*ones(1,Nk-1));
% 
%     end
% 
%    
% 
%     for t = 1:(Ns-Nt)
% 
%         Test_Data_RNN{end+1}  = user_data(:,:,1,Test_set(t));
% 
%         Test_Label_RNN{end+1} = categorical(u*ones(1,Nk-1));
% 
%     end   
% 
%        
% 
%     Train_Label_CNN = cat(1,Train_Label_CNN,categorical(u*ones(Nt,1)));
% 
%     Test_Label_CNN = cat(1,Test_Label_CNN,categorical(u*ones(Ns-Nt,1)));
% 
% end
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% %%% Standard Mahalanobis Classifier
% 
% 
% Train_Data = Train_Data_CNN;
% 
% Test_Data = Test_Data_CNN;
% 
% 
% Train_count = size(Train_Data,4);
% 
% Test_count = size(Test_Data,4);
% 
% 
% Test_Data = (Test_Data - repmat(mean(Train_Data,4),1,1,1,Test_count))./(repmat(std(Train_Data,[],4),1,1,1,Test_count));
% 
% Train_Data = (Train_Data - repmat(mean(Train_Data,4),1,1,1,Train_count))./(repmat(std(Train_Data,[],4),1,1,1,Train_count));
% 
% 
% Train_Label = Train_Label_CNN;
% 
% Test_Label = Test_Label_CNN;
% 
% 
% Users = double(unique(Test_Label)); Users = Users'; N_users = length(Users);
% 
% 
% for u = Users
% 
%    Templates.mean(:,:,u) = mean(Train_Data(:,:,1,ismember(double(Train_Label),u)),4);
% 
%    Templates.std(:,:,u) = std(Train_Data(:,:,1,ismember(double(Train_Label),u)),[],4);  
% 
% end
% 
% 
% considered_features = 1:8;
% 
% genuine_scores = []; impostor_scores = [];
% 
% for t = 1:Test_count
% 
%    
% 
%     %%% Identification   
% 
%     sample = Test_Data(:,:,1,t); us = double(Test_Label(t)); imps = setdiff(Users,us);
% 
%     scores = sum(((repmat(sample,1,1,N_users) - Templates.mean).^2)./(Templates.std.^2),2); scores = permute(scores,[1 3 2]);
% 
%     [~,index] = min(scores,[],2); ids(t) = mode(index(considered_features)); Res(t) = ids(t) == us;
% 
%    
% 
%     %%% Verification
% 
%     genuine_scores(:,end+1) = scores(:,us);
% 
%     impostor_scores(:,end+1:end+length(imps)) = scores(:,imps);
% 
% end
% 
% Accuracy = sum(Res)/Test_count;
% 
% 
% scs = genuine_scores(:); scs = sort(scs,'ascend');
% 
% MIN = scs(0.03*length(scs)); MAX = scs(0.97*length(scs));
% 
% 
% Thresholds = MIN:(MAX - MIN)/199:MAX;
% 
% 
% considered_features = 1:8;
% 
% genuine_scores_f = genuine_scores(considered_features,:);
% 
% impostor_scores_f = impostor_scores(considered_features,:);
% 
% 
% for h = 1:200
% 
%    for f = 1:length(considered_features)
% 
%        frr_outcomes(f,:,h) = genuine_scores(f,:) >= Thresholds(h);
% 
%        far_outcomes(f,:,h) = impostor_scores(f,:) < Thresholds(h);
% 
%    end
% 
%    total_frr_outcomes(h,:) = mode(frr_outcomes(:,:,h),1);
% 
%    total_far_outcomes(h,:) = mode(far_outcomes(:,:,h),1);
% 
%    FRR(h) = sum(total_frr_outcomes(h,:),2)/size(total_frr_outcomes,2);
% 
%    FAR(h) = sum(total_far_outcomes(h,:),2)/size(total_far_outcomes,2);
% 
% end
% 
% 
% Diff = FRR - FAR; i = 1; val = Diff(i);
% 
% while val > 0
% 
%     i = i + 1;
% 
%     val = Diff(i);
% 
% end
% 
% EER = (FRR(i) + FAR(i))/2;
% 
% EER_Thresh = Thresholds(i);
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%
% 
% 
%  
% 
% %%% v2
% 
% 
% Train_Data = Train_Data_CNN;
% 
% Test_Data = Test_Data_CNN;
% 
% 
% Train_count = size(Train_Data,4);
% 
% Test_count = size(Test_Data,4);
% 
% 
% Test_Data = (Test_Data - repmat(mean(Train_Data,4),1,1,1,Test_count))./(repmat(std(Train_Data,[],4),1,1,1,Test_count));
% 
% Train_Data = (Train_Data - repmat(mean(Train_Data,4),1,1,1,Train_count))./(repmat(std(Train_Data,[],4),1,1,1,Train_count));
% 
% 
% Train_Label = Train_Label_CNN;
% 
% Test_Label = Test_Label_CNN;
% 
% 
% Users = double(unique(Test_Label)); Users = Users'; N_users = length(Users);
% 
% 
% for u = 1:Users
% 
%     Templates_v2.mean(:,u) = mean(mean(Train_Data(:,:,1,ismember(double(Train_Label),u)),4),2);
% 
%     for f = 1:size(Train_Data,1)
% 
%         vals = Train_Data(f,:,1,ismember(double(Train_Label),u));
% 
%         Templates_v2.std(f,u) = std(vals(:));
% 
%     end
% 
% end
% 
% 
% considered_features = 1:8;
% 
% 
% Res = []; ids = [];
% 
% for t = 1:Test_count
% 
%      for f = 1:size(Test_Data,1)
% 
%           for n = 1:size(Test_Data,2)
% 
%               scores = ((repmat(Test_Data(f,n,1,t),1,150)- Templates_v2.mean(f,:)).^2)./(Templates_v2.std(f,:).^2);
% 
%               [~,index] = min(scores); ids(f,n) = index(1);               
% 
%           end
% 
%      end
% 
%      r = ids(considered_features,:);
% 
%      Res(t) = mode(r(:)) == double(Test_Label(t));
% 
% end
% 
% Accuracy = sum(Res)/Test_count;