%% Lambda Sweeping for Selected Systems

nelx = 100; nely = 20; nelz = 20;
volfrac = 0.3; penal = 3; rmin = 1.5;

global ldx ldxy ldxyz KE iindex jindex
global Mats Rhs Initgs

cut = 0.1; U = []; eps = 1e-3;
change = 1.;
ldx = nelx+1;
ldxy = (nely+1)*ldx;
ldxyz = (nelz+1)*ldxy;
[KE] = lk; LKE = reshape(KE,24*24,1);
[iindex,jindex]=sparsitypattern(nelx,nely,nelz);

% DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
F = sparse(3*ldxyz,1);
F(dof(nelx+1,1,[1:nelz+1],2),1) = -1;
fixeddofs   = dof(1,[1:nely+1],[1:nelz+1],[1:3]);
alldofs     = [1:3*ldxyz];
freedofs    = setdiff(alldofs,fixeddofs);
U = sparse(3*ldxyz,1);

phase = 1:6;

disp("Solving systems 80-86...")

filename = sprintf('rho_iter_large_80.mat');
data = load(filename, 'x');
xold = data.x;
fprintf(1, 'Old System %d\n', 80);

[vold,~] = ComputeNnz(nelx,nely,nelz,penal,xold,xold,cut,eps,jindex);
Ktrue=sparse(iindex,jindex,vold);
Atrue=Ktrue(freedofs,freedofs);
b = F(freedofs,:);
tc = 1:1:length(freedofs);

display('computing preconditioner...');
ilu_options.type = 'ilutp';
ilu_options.droptol = 1.e-4;
[M1,M2] = ilu(Atrue,ilu_options);
P_funright = @(z) M2 \ (M1 \ z);

% %------ Get initial solution -------------
if (isempty(U))
    % x0_hat = zeros(length(freedofs),1);
    x0 = zeros(length(freedofs),1);
    U = sparse(3*ldxyz,1);
    recyclespace = []; last = 0;
else
    % x0_hat = D*U(freedofs,1);
     x0 = U(freedofs,1);
end

% Solving first system
restart = 50;
tol = 1.e-8;
maxit = 1001;
map = speye(size(Atrue,2));
x0 = zeros(size(Atrue,2),1);
mats = {Atrue,map};
P_funright = @(z) M2 \ (M1 \ z);
[sol, ~, r_nrm,~,~,~] = pmgmres(mats, x0, b, P_funright, tol, maxit, restart); % Replaced with Eric's GMRES
U(freedofs,:) = sol;
U(fixeddofs,:)= 0;

Aold = []; Patt = speye(size(Atrue,2));
changedCols = []; cc = [];

% Solving the first 6 systems
for i = 1:length(phase)
    filename = sprintf('rho_iter_large_%d.mat',80+i);
    data = load(filename, 'x');
    x = data.x;
    fprintf(1, 'Current System %d\n', 80+i);

    % Forming Current system
    tc = 1:1:length(freedofs);
    [v, uc] = ComputeNnz(nelx,nely,nelz,penal,x,xold,cut,eps,jindex);

    display('generating matrix...');
    K=sparse(iindex,jindex,v);
    A=K(freedofs,freedofs);

    % Pruning the set of selected columns
    NewCols = setdiff(uc,changedCols);
    [nf, nloc] = ismember(NewCols,freedofs); 
    nc = tc(nloc(nf));
    fprintf(1, 'Number of new Cols : %d\n', length(nc));

    RudCols = setdiff(changedCols,uc);
    rc = tc(ismember(freedofs,RudCols));
    fprintf(1, 'Number of removed Cols : %d\n', length(rc));
        
    [tf, loc] = ismember(changedCols, freedofs); 
    cc = tc(loc(tf));

    if ~isempty(rc)
        % Removing columns of Aold
        changedCols = setdiff(changedCols,RudCols,"stable");
        mask = ismember(cc,rc); cc = cc(~mask); 
        changedCols = cat(1,changedCols,NewCols);
        cc = [cc,nc];
    else
        changedCols = cat(1,changedCols,NewCols);
        cc = [cc,nc];
        mask = [];
    end

    % Mapping from cols to elements
    [Inds,Jnds,Vals] = colsToDofs(NewCols,xold,penal,LKE);
    Rows = Inds(:); Rows = Rows(Rows~=0); 
    Cols = Jnds(:); Cols = Cols(Cols~=0);
    Val = Vals(:); Val = Val(Val~=0); 

    % Assembling new columns
    Rows = [Rows; max(freedofs)]; Cols = [Cols;max(NewCols)];
    Val = [Val; 0];
    Kcols = sparse(Rows,Cols,Val);
    Newdofs = freedofs(nloc(nf));
    Acols = Kcols(freedofs,Newdofs);

    Aold(:,mask) = [];
    Aold = cat(2,Aold,Acols);
    
    % Sparsifying with ADMM
    [lambda, ~, rho, ColInd, ~] = LambdaLineSearch(A,Acols,nc);
    fprintf(1, 'lambda val selected as : %d\n', lambda);
    fprintf(1, 'rho val selected as : %d\n', rho);
    fprintf(1, 'Column Ind selected: %d\n', ColInd);
    [PattNew, Resi, ~] = SAML1(A, Acols, nc, lambda, rho, 1);
    Patt = Patt + PattNew;
    disp(['Number of non-zeros of SAM pattern: ', num2str(nnz(Patt))]);

    % Computing mapping
    map = SAM(A,Aold,cc,Patt);
    Err = norm(A*map-Atrue, 'fro')/norm(Atrue, 'fro');
    disp(['Rel Accuracy of mapping: ', num2str(Err)]);

     % %------ Get initial solution -------------
     if (isempty(U))
         % x0_hat = zeros(length(freedofs),1);
         x0 = zeros(length(freedofs),1);
         U = sparse(3*ldxyz,1);
         recyclespace = []; last = 0;
     else
         % x0_hat = D*U(freedofs,1);
         x0 = U(freedofs,1);
     end
    
    % Solving first system
    restart = 50;
    tol = 1.e-8;
    maxit = 1001;
    mats = {A,map};
    P_funright = @(z) M2 \ (M1 \ z);
    [sol, ~, r_nrm,~,~,~] = pmgmres(mats, x0, b, P_funright, tol, maxit, restart); % Replaced with Eric's GMRES
    U(freedofs,:) = sol;
    U(fixeddofs,:)= 0;

end

% Solving systems 87, 88, ...., 93 in succession
Sys = 87:1:93;

Iter = cell(length(Sys),1);
SAMAcc = cell(length(Sys),1);
NCAcc = cell(length(Sys),1);
PriAcc = cell(length(Sys),1);
Spar = cell(length(Sys),1);
LambdaVals = cell(length(Sys),1);
IdealLambda = zeros(length(Sys),1);
Nonzeros{i} = cell(length(Sys),1);

for i = 1:length(Sys)
    changedColsN = changedCols;
    ccn = cc;
    Aoldn = Aold;
    Pattn = Patt;

    for k = 1:(Sys(i)-Sys(1)+1)
        % Old Systems
        filename = sprintf('rho_iter_large_%d.mat',87+k-1);
        data = load(filename, 'x');
        x = data.x;
        fprintf(1, 'Current System %d\n', 87+k-1);

        % Assembling old systems
        [v, uc] = ComputeNnz(nelx,nely,nelz,penal,x,xold,cut,eps,jindex);

        display('Re-generating old matrix...');
        K=sparse(iindex,jindex,v);
        A=K(freedofs,freedofs);

        % Pruning the set of selected columns
        NewCols = setdiff(uc,changedColsN);
        [nf, nloc] = ismember(NewCols,freedofs); 
        nc = tc(nloc(nf));
        fprintf(1, 'Number of new Cols : %d\n', length(nc));

        RudCols = setdiff(changedColsN,uc);
        rc = tc(ismember(freedofs,RudCols));
        fprintf(1, 'Number of removed Cols : %d\n', length(rc));
        
        [tf, loc] = ismember(changedColsN, freedofs); 
        ccn = tc(loc(tf));
        
        if ~isempty(rc)
            % Removing columns of Aold
            changedColsN = setdiff(changedColsN,RudCols,"stable");
            mask = ismember(ccn,rc); ccn = ccn(~mask); 
            changedColsN = cat(1,changedColsN,NewCols);
            ccn = [ccn,nc];
        else
            changedColsN = cat(1,changedColsN,NewCols);
            ccn = [ccn,nc];
            mask = [];
        end

        % Mapping from cols to elements
        [Inds,Jnds,Vals] = colsToDofs(NewCols,xold,penal,LKE);
        Rows = Inds(:); Rows = Rows(Rows~=0); 
        Cols = Jnds(:); Cols = Cols(Cols~=0);
        Val = Vals(:); Val = Val(Val~=0); 

        % Assembling new columns
        Rows = [Rows; max(freedofs)]; Cols = [Cols;max(NewCols)];
        Val = [Val; 0];
        Kcols = sparse(Rows,Cols,Val);
        Newdofs = freedofs(nloc(nf));
        Acolsn = Kcols(freedofs,Newdofs);

        Aoldn(:,mask) = [];
        Aoldn = cat(2,Aoldn,Acolsn);
    
        % Sparsifying with ADMM
        [lambda, ~, rho, ColInd, ~] = LambdaLineSearch(A,Acolsn,nc);
        fprintf(1, 'lambda val selected as : %d\n', lambda);
        fprintf(1, 'rho val selected as : %d\n', rho);
        fprintf(1, 'Column Ind selected: %d\n', ColInd);

        % Solving all prior systems 
        if k < Sys(i)-Sys(1)+1
            [PattNew, Resi, ~] = SAML1(A, Acolsn, nc, lambda, rho, 1);
            Pattn = Pattn + PattNew;

            % Computing mapping
            map = SAM(A,Aoldn,ccn,Patt);
            disp(['Number of non-zeros of SAM pattern: ', num2str(nnz(Pattn))]);

            % %------ Get initial solution -------------
            if (isempty(U))
                % x0_hat = zeros(length(freedofs),1);
                x0 = zeros(length(freedofs),1);
                U = sparse(3*ldxyz,1);
                recyclespace = []; last = 0;
            else
                % x0_hat = D*U(freedofs,1);
                x0 = U(freedofs,1);
            end

            % Solving first system
            restart = 50;
            tol = 1.e-8;
            maxit = 1001;
            mats = {A,map};
            P_funright = @(z) M2 \ (M1 \ z);
            [sol, ~, r_nrm,~,~,~] = pmgmres(mats, x0, b, P_funright, tol, maxit, restart); % Replaced with Eric's GMRES
            U(freedofs,:) = sol;
            U(fixeddofs,:)= 0;

        else
            [lambda, Lambdas, rho, ColInd, ~] = LambdaLineSearch(A,Aoldn,ccn);

            SAMErr = zeros(length(Lambdas),1);
            PriErr = zeros(length(Lambdas),1);
            Cnnz = zeros(length(Lambdas),1);
            iter = zeros(length(Lambdas),1);
            Zeors = zeros(length(Lambdas),1);
            Nnzs = cell(length(Lambdas),1);

            % Lambda Sweeping
            fprintf(1, 'Lambda Sweeping with system: %d\n', Sys(i))

            for j = 1:length(Lambdas)
                [PattNew, Resi, ~] = SAML1(A, Aoldn, ccn, Lambdas(j), rho, 1);
                PattAlt = speye(size(A,2)) + PattNew;
                PriErr(j) = norm(Resi,2)/norm(Atrue(:,ccn),'fro');
                Cnnz(j) = nnz(PattNew(:,ccn))/length(ccn);
                
                % Computing mapping
                map = SAM(A,Aoldn,ccn,PattAlt);
                SAMErr(j) = norm(A*map(:,ccn)-Atrue(:,ccn), 'fro')/norm(Atrue(:,ccn), 'fro');
                
                Nnzs{j} = [ccn; sum(PattAlt(:,ccn), 1)];
 
                % %------ Get initial solution -------------
                if (isempty(U))
                    % x0_hat = zeros(length(freedofs),1);
                    x0 = zeros(length(freedofs),1);
                    U = sparse(3*ldxyz,1);
                    recyclespace = []; last = 0;
                else
                    % x0_hat = D*U(freedofs,1);
                    x0 = U(freedofs,1);
                end
                
                % Solving first system
                restart = 50;
                tol = 1.e-8;
                maxit = 1001;
                mats = {A,map};
                P_funright = @(z) M2 \ (M1 \ z);
                [sol, ~, r_nrm,iter(j),~,~] = pmgmres(mats, x0, b, P_funright, tol, maxit, restart); % Replaced with Eric's GMRES
    
            end

        end

    end

    Iter{i} = iter;
    SAMAcc{i} = SAMErr;
    NCAcc{i} = ccn;
    PriAcc{i} = PriErr;
    Spar{i} = Cnnz;
    LambdaVals{i} = Lambdas;
    IdealLambda(i) = lambda;
    Nonzeros{i} = Nnzs;

end

save LateSweepLarge Iter SAMAcc NCAcc PriAcc Spar LambdaVals IdealLambda Nonzeros


%% Support functions 
function [KE]=lk
E=1.0;
nu=0.3;
t=E/(1+nu)/(1-2*nu);
ind=[[1,2,2,3,4,4,5,-4,6,7,-2,8,5,6,-4,7,8,-2,9,-8,-8,10,-6,-6],
    [2,1,2,-4,5,6,4,3,4,-2,7,8,6,5,-4,-8,9,-8,8,7,-2,-6,10,-6],
    [2,2,1,-4,6,5,6,-4,5,-8,-8,9,4,4,3,-2,8,7,8,-2,7,-6,-6,10],
    [3,-4,-4,1,-2,-2,7,2,-8,5,4,-6,7,-8,2,5,-6,4,10,6,6,9,8,8],
    [4,5,6,-2,1,2,2,7,8,-4,3,4,8,9,-8,-6,5,-4,6,10,-6,-8,7,-2],
    [4,6,5,-2,2,1,8,-8,9,-6,-4,5,2,8,7,-4,4,3,6,-6,10,-8,-2,7],
    [5,4,6,7,2,8,1,-2,2,3,-4,4,9,8,-8,10,6,-6,5,-6,-4,7,-8,-2],
    [-4,3,-4,2,7,-8,-2,1,-2,4,5,-6,-8,7,2,6,10,6,-6,5,4,8,9,8],
    [6,4,5,-8,8,9,2,-2,1,-4,-6,5,8,2,7,-6,6,10,4,-4,3,-2,-8,7],
    [7,-2,-8,5,-4,-6,3,4,-4,1,2,-2,10,-6,6,9,-8,8,7,8,2,5,6,4],
    [-2,7,-8,4,3,-4,-4,5,-6,2,1,-2,-6,10,6,8,7,2,-8,9,8,6,5,4],
    [8,8,9,-6,4,5,4,-6,5,-2,-2,1,6,6,10,-8,2,7,2,-8,7,-4,-4,3],
    [5,6,4,7,8,2,9,-8,8,10,-6,6,1,2,-2,3,4,-4,5,-4,-6,7,-2,-8],
    [6,5,4,-8,9,8,8,7,2,-6,10,6,2,1,-2,-4,5,-6,4,3,-4,-2,7,-8],
    [-4,-4,3,2,-8,7,-8,2,7,6,6,10,-2,-2,1,4,-6,5,-6,4,5,8,8,9],
    [7,-8,-2,5,-6,-4,10,6,-6,9,8,-8,3,-4,4,1,-2,2,7,2,8,5,4,6],
    [8,9,8,-6,5,4,6,10,6,-8,7,2,4,5,-6,-2,1,-2,2,7,-8,-4,3,-4],
    [-2,-8,7,4,-4,3,-6,6,10,8,2,7,-4,-6,5,2,-2,1,-8,8,9,6,4,5],
    [9,8,8,10,6,6,5,-6,4,7,-8,2,5,4,-6,7,2,-8,1,-2,-2,3,-4,-4],
    [-8,7,-2,6,10,-6,-6,5,-4,8,9,-8,-4,3,4,2,7,8,-2,1,2,4,5,6],
    [-8,-2,7,6,-6,10,-4,4,3,2,8,7,-6,-4,5,8,-8,9,-2,2,1,4,6,5],
    [10,-6,-6,9,-8,-8,7,8,-2,5,6,-4,7,-2,8,5,-4,6,3,4,4,1,2,2],
    [-6,10,-6,8,7,-2,-8,9,-8,6,5,-4,-2,7,8,4,3,4,-4,5,6,2,1,2],
    [-6,-6,10,8,-2,7,-2,8,7,4,4,3,-8,-8,9,6,-4,5,-4,6,5,2,2,1]];

p=[4/9-2*nu/3, 1/12, -1/9, nu/3-1/12, 1/18, 1/24, -5/36+nu/6, ...
    nu/6-1/24,-1/18+nu/6,-1/9+nu/6];
%p=[4/9-2*nu/3,    1/12,    1/18,    1/24,    1/12-nu/3, ...
%   -1/18+nu/6,    1/24-nu/6,  -1/9,   -5/36+nu/6,  -1/9+nu/6];
for i=1:24
    for j=1:24
        if (ind(i,j)>0)
            KE(i,j) = p(ind(i,j))*t;
        else
            KE(i,j) = -p(-ind(i,j))*t;
        end
    end
end
end 

function s=dof(xs,ys,zs,ds)
global ldx ldxy ldxyz
lx=length(xs);
ly=length(ys);
lz=length(zs);
count = 1;
s=zeros(24,1);
for iz=1:lz
    for iy=1:ly
        for ix=1:lx
            p=((zs(iz)-1)*ldxy+(ys(iy)-1)*ldx+(xs(ix)-1))*3;
            s(count:count+2)=p+ds;
            count = count+3;
        end
    end
end
end

function [i,j]=sparsitypattern(nelx,nely,nelz)
global ldxyz ldxy ldx
ind = 0;
i = zeros(24*24*nelx*nely*nelz,1);
j = zeros(24*24*nelx*nely*nelz,1);
for elz = 1:nelz
    for ely = 1:nely
        for elx = 1:nelx
            el_dofs = dof([elx:elx+1],[ely:ely+1],[elz:elz+1],[1:3]);

            for p=1:24
                i([1:24]+ind)=el_dofs(p);
                j([1:24]+ind)=el_dofs;
                ind = ind + 24;
            end
        end
    end
end
end

function [v,uc] = ComputeNnz(nelx,nely,nelz,penal,x,xold,cut,eps,jindex)
    global KE
    uc = [];
    v = zeros(24*24*nelx*nely*nelz,1);
    ind = 0;
    LKE = reshape(KE,24*24,1); 
    for elz = 1:nelz
        for ely = 1:nely
            for elx = 1:nelx
                if (abs(x(elx,ely,elz)) > eps) && (abs(x(elx,ely,elz)-xold(elx,ely,elz))/abs(x(elx,ely,elz)) >= cut)
                    % Identifying Changing columns
                    C = jindex(ind+[1:24*24]); 
                    uc = [uc;unique(C)];
                    uc = unique(uc);
                end
                v(ind+[1:24*24])=x(elx,ely,elz)^penal*LKE;
                ind=ind+24*24;
            end
        end
    end

end

function [Inds, Jnds, Vals] = colsToDofs(uc,xold,penal,LKE)
    global ldx ldxy
    Inds = zeros(81,length(uc)); Jnds = Inds;
    Vals = zeros(81,length(uc));
    KE = reshape(LKE,24,24);
    % nold = -1; cc = 1;
    ds = [1:3];

    for k = 1:length(uc)
        J = uc(k); s = J-[1:3];
        m = mod(s,3)==0;
        n = s(m)/3; % node number
        cc = ds(m);

        % Finding neighboring nodes
        nodes = [0, 1, -1, ldx, ldx+1, ldx-1, ldxy, ldxy+1, ldxy-1, ldxy+ldx, ...
            ldxy+ldx+1, ldxy+ldx-1, ldxy-ldx, ldxy-ldx+1, ldxy-ldx-1, -ldx, -ldx+1, -ldx-1, ...
            -ldxy, -ldxy+1, -ldxy-1, -ldxy-ldx, -ldxy-ldx-1, -ldxy-ldx+1, -ldxy+ldx, -ldxy+ldx-1, ...
             -ldxy+ldx+1];

        % Computing associated element
        nz = floor(n/ldxy)+1;
        if nz < 12
            ny = floor((n-(nz-1)*ldxy)/ldx)+1;
            if ny < 12
                nx = n-(nz-1)*ldxy-(ny-1)*ldx+1;
            end
        end

        % Identifying node type and filtering neighbors
        % [neighbors,b,nx,ny,nz] = ncol(n,nodes,nx,ny,nz,ldx,ldxy);
        % [neighbors,b,nx,ny,nz] = ncolSmall(n,nodes,nx,ny,nz,ldx,ldxy);
        [neighbors,b,nx,ny,nz] = ncolLarge(n,nodes,nx,ny,nz,ldx,ldxy);

        % Computing row indices
        Row = zeros(81,1);

        s = 3 * neighbors(:)' + (1:3)';
        Row(1:numel(s)) = s(:);
        
        % Catching repeated elements
        % if n == nold
        %     cc = cc + 1; 
        % else
        %     cc = 1; 
        % end

        % Contribution from neighboring elements
        Nz = nvals(xold,b,penal,KE,cc,nx,ny,nz);
       
        % nold = n;
        Inds(1:nnz(Nz),k) = Row(Nz~=0); % Non-zeros per column
        Jnds(1:nnz(Nz),k) = uc(k).*ones(nnz(Nz),1);
        Vals(1:nnz(Nz),k) = Nz(Nz~=0); % Non-zero entries
    end
end

function [m, val, breaks] = admmLasso(G, b, rho, lambda, maxIter)
    % Parameters: 
    % abstol: e.g., 1e-5
    % reltol: e.g., 1e-3
    % mu = 10, tau = 2

    [~, p] = size(G);
    
    % Initialize variables
    z = zeros(p, 1);
    u = zeros(p, 1);

    % Precompute using Cholesky (faster for symmetric positive definite)
    G_stack = G' * G + rho * eye(p);
    L = chol(G_stack, 'lower'); 
    Gty = G' * b;

    % Absolute and relative epsilon
    epsAbs = 1e-5;
    epsRel = 1e-3;

    % Dynamic penalty parameters
    mu = 1; tau = 1;
    
    for i = 1:maxIter      
        % 1. m-update: Solve (G'G + rho*I)m = G'b + rho(z - u)
        % Using two-step back/forward substitution with Cholesky
        rhs = Gty + rho * (z - u);
        m = L' \ (L \ rhs);
        
        % 2. z-update (Soft Thresholding)
        kappa = lambda / rho;
        val = m + u;
        zold = z; % Preserving previous z 
        z = sign(val) .* max(abs(val) - kappa, 0);

         % 3. u-update (Dual Update)
         u = u + m - z;
        
        % Updating penalty parameter 
        r = m-z; Priresi = norm(r,2);
        s = rho*(z-zold); Dualresi = norm(s,2);
        
        % if Priresi > mu*Dualresi
        %     rho = rho*tau;
        %     u = u/tau;
        % elseif Dualresi > mu*Priresi
        %     rho = rho/tau;
        %     u = u*tau;
        % end

        % Adaptive stopping criterion
        epsPri = epsAbs + epsRel*max([norm(m,2), norm(z,2)]);
        epsDual = epsAbs + epsRel*rho*norm(u,2);
        if (Dualresi<epsDual) && (Priresi<epsPri)
            % disp(["Breaking at iteration i = ", num2str(i)]);
            % disp(["Penalty parameter value rho = ", num2str(rho)]);
            breaks = i;
            break;

        else 
            breaks = i;
        end

    end

    m = z;
end
