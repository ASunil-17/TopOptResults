function [N, Resi, Zs] = SAML1(A, A0, uniqueCols, lambda, rho, maxIter, varargin)
    
    % For constructing the SAM between a sequence of systems. Methodology
    % based on algorihtms 4.1 and 4.2 from Preconditioning parametrized 
    % linear systems by Carr, de Sturler, and Gugercin.  
    % Output: The mapping between 
    % Ak = Systems(k).Systems and A0 = Systems(1).Systems. Count is the
    % number of non-zeros in mapping.
    % Input: Systems(k).systems. Must be of the same dimensions.
    % This function is replacing the least-squares solve with L1
    % regularized least-squares solve. The goal is to extract the pattern
    % that emerges.

    % Parsing input
    p = inputParser;
    p.CaseSensitive = false;
    empty_matrix_flag = {''}; 

    addRequired(p, 'A', @(x) issparse(x) && ismatrix(x));
    addRequired(p, 'A0', @(x) issparse(x) && ismatrix(x));
    addRequired(p, 'uniqueCols');
    addRequired(p, 'lambda');
    addRequired(p, 'maxIter');
    addOptional(p, 'PP', empty_matrix_flag, @(x) ismatrix(x));
    parse(p, A, A0, uniqueCols, lambda, maxIter, varargin{:});
    PP = p.Results.PP;

    if isequal(PP, empty_matrix_flag)
        % PP was not provided; handle the default case (e.g., set to Patt A0)
        fprintf('Optional matrix PP was NOT provided. Using a default.\n');

        % Using sparsity patterns diag
        %[I,J] = find(diag(diag(A0)));
        %findA = sparse(I,J,1);
        %PP = logical(findA);

        [I,J] = find(A);
        findA = sparse(I,J,1);
        PP2 = logical(findA);
        PP = PP2;
        PP3 = logical(sparse(double(PP2)*double(PP)));
        nnzMM = nnz(PP3);
    else 
        % PP was provided; check its dimensions are compatible
        if ~all(size(PP) == size(A))
             error('SAM:InvalidPPSize', 'Optional matrix PP must have the same dimensions as A.');
        end

        [I,J] = find(A);
        findA = sparse(I,J,1);
        PP2 = logical(findA);
        PP3 = logical(sparse(double(PP2)*double(PP)));
        nnzMM = nnz(PP3);
    end 

    % Computing mapping from identity
    N = speye(size(A,1));

    % Pre-allocation
    nnz_M = zeros(1,length(uniqueCols));
    nnz_LS = zeros(1,length(uniqueCols));

    % Pre-processing
    for k = 1:length(uniqueCols)
        nnz_M(k) = nnz(PP(:,uniqueCols(k)));
        nnz_LS(k) = nnz(PP3(:,uniqueCols(k)));
    end

    max_col = max(nnz_M);
    max_row = max(nnz_LS);
    G = zeros(max_row,max_col);
    Zs = zeros(length(uniqueCols),1);

    for k = 1:length(uniqueCols)
        row = find(PP3(:,uniqueCols(k))); % Finding non-zeros in kth column of PP3
        col = find(PP(:,uniqueCols(k))); % Finding non-zeros in kth column of PP
        G(1:nnz_LS(k),1:nnz_M(k)) = A(row,col);
        % if uniqueCols(k) == 17424
        %     fprintf(1, "Probing ADMM for col %d\n", uniqueCols(k));
        % end
        [SM, ~] = admmLasso(G(1:nnz_LS(k), 1:nnz_M(k)), A0(row,k), rho, lambda, maxIter);
        if ~any(SM(:))
            Zs(k) = 1;
        end
        Resi(k) = norm(A0(row,k)-G(1:nnz_LS(k), 1:nnz_M(k))*SM);
        col = col(find(SM)); nnz_M(k) = nnz(SM);
        N(col, uniqueCols(k)) = ones(nnz_M(k),1);
    end

end 

function [m, breaks] = admmLasso(G, b, rho, lambda, maxIter)
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
        if rho == 0
            kappa = lambda;
        else
            kappa = lambda / rho;
        end
        
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
