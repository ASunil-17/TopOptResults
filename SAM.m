%% SAM function

function N = SAM(A, A0, uniqueCols, varargin)
    
    % For constructing the SAM between a sequence of systems. Methodology
    % based on algorihtms 4.1 and 4.2 from Preconditioning parametrized 
    % linear systems by Carr, de Sturler, and Gugercin.  
    % Output: computes the row, col, and val indices for a mapping between 
    % Ak = Systems(k).Systems and A0 = Systems(1).Systems. Count is the
    % number of non-zeros in mapping.
    % Input: Systems(k).systems. Must be of the same dimensions.

    % Parsing input
    p = inputParser;
    p.CaseSensitive = false;
    empty_matrix_flag = {''}; 

    addRequired(p, 'A', @(x) issparse(x) && ismatrix(x));
    addRequired(p, 'A0', @(x) issparse(x) && ismatrix(x));
    addRequired(p, 'uniqueCols');
    addOptional(p, 'PP', empty_matrix_flag, @(x) ismatrix(x));
    parse(p, A, A0, uniqueCols, varargin{:});
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
        PP = logical(findA);
        PP3 = logical(sparse(double(PP)*double(PP)));
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
    end 

    % Constructing mapping from identity
    N = speye(size(A,1));

    % Pre-allocation
    nnz_M = zeros(1,length(uniqueCols));
    nnz_LS = zeros(1,length(uniqueCols));
  
    % Computing SAM 
    for k = 1:length(uniqueCols)
        % Initialize these later for efficiency...
        nnz_M(k) = nnz(PP(:,uniqueCols(k)));
        nnz_LS(k) = nnz(PP3(:,uniqueCols(k)));
    end

    max_col = max(nnz_M);
    max_row = max(nnz_LS);
    G = zeros(max_row,max_col);

    for k = 1:length(uniqueCols)
        row = find(PP3(:,uniqueCols(k))); % Finding non-zeros in kth column of PP3
        col = find(PP(:,uniqueCols(k))); % Finding non-zeros in kth column of PP
        G(1:nnz_LS(k),1:nnz_M(k)) = A(row,col);
        N(col,uniqueCols(k)) = G(1:nnz_LS(k),1:nnz_M(k))\A0(row,k);
    end

end 