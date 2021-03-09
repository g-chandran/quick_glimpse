from modules import *
from sklearn.decomposition import PCA
from pulp import *
import warnings
warnings.filterwarnings("ignore")


class IP_summarizer:
    def __init__(self, input_document, document_matrix, threshold=80, max_len=35):
        '''
        :param input_document: sequence of sentences numpy array
        :param document_matrix: (T,dim) 2D numpy array
        :param threshold:
        '''
        self.input_document = input_document
        self.document_matrix = document_matrix
        self.threshold = threshold
        self.max_len = max_len

        if len(self.document_matrix) == 1:
            self.N = 1
            self.selected_id = [0]
            self.obj = 0
        else:
            self.do_PCA()

    def number_of_SummSent(self, var_ratios):
        sum_ = 0
        for i, v in enumerate(var_ratios):
            if sum_ >= self.threshold:
                return i
            else:
                sum_ += v

    def do_PCA(self):
        '''
        :param Doc_mat: (T,dim) 2d array
        :return:
        '''
        pe = positional_encoding(self.document_matrix)
        self.Doc_mat_ = self.document_matrix + pe

        for k in range(1):
            self.Doc_mat_ = attention(
                self.Doc_mat_, self.Doc_mat_, self.Doc_mat_)

        tr_Doc_mat_ = np.transpose(self.Doc_mat_)

        pca = PCA()
        pca.fit(tr_Doc_mat_)
        self.var_ratio = pca.explained_variance_ratio_ * 100

        self.N = self.number_of_SummSent(self.var_ratio)
        self.N = min(self.N, 7)
        self.PCs = pca.fit_transform(tr_Doc_mat_)[:, :self.N]
        self.alpha_M = pca.components_
        print("\nPCA: " + str(self.alpha_M))

    def sent_importance(self, Doc_mat):
        '''
        :param Doc_mat: (T,dim) 2d array
        :param PCs: ()
        :param var_ratio:
        :return:
        '''
        sq_D = np.square(Doc_mat)
        sq_D_sum = 1 / np.sqrt(np.sum(sq_D, 1))

        sq_P = np.square(self.PCs)
        sq_P_sum = 1 / np.sqrt(np.sum(sq_P, 0))

        normalize_factor = np.matmul(np.reshape(
            sq_D_sum, [-1, 1]), np.transpose(np.reshape(sq_P_sum, [-1, 1])))

        unnormalized_similarity = np.matmul(Doc_mat, self.PCs)
        normalized_similarity = np.multiply(
            normalize_factor, unnormalized_similarity)
        weighted_similarity = (
            self.var_ratio[:self.N]/100) * np.exp(normalized_similarity)

        importance = np.mean(weighted_similarity, 1)
        print("\nSentence Importance: " + str(importance))
        return importance

    def sent_importance_loading(self):
        '''
        :param PCs: ()
        :param var_ratio:
        :return:
        '''

        importance = []
        v_ref = np.array([1 / np.sqrt(len(self.var_ratio))]
                         * len(self.var_ratio))
        for i in range(len(self.var_ratio)):
            imp = np.sum(self.var_ratio *
                         np.maximum((np.abs(self.alpha_M[:, i]) - v_ref), 0))
            importance.append(imp)

        return np.array(importance)

    def pairwise_sent_similarity(self, Doc_mat):
        sq = np.square(Doc_mat)
        sq_sum = 1 / np.sqrt(np.sum(sq, 1))
        normalize_factor = np.matmul(np.reshape(
            sq_sum, [-1, 1]), np.transpose(np.reshape(sq_sum, [-1, 1])))
        sim_matrix_ = np.matmul(Doc_mat, np.transpose(Doc_mat))
        sim_matrix = np.multiply(normalize_factor, sim_matrix_)
        return sim_matrix

    def return_similarity_list(self, sim_matrix):
        sims = []
        for i, v in enumerate(sim_matrix):
            if i == len(sim_matrix) - 1:
                continue
            else:
                sims.append(sim_matrix[i][i + 1:].tolist())

        def flatten(l): return [item for sublist in l for item in sublist]
        sims = flatten(sims)
        return sims

    def Optimization(self):
        imp = self.sent_importance(self.Doc_mat_)

        similarity = self.pairwise_sent_similarity(self.Doc_mat_)

        if len(imp) > self.max_len:
            ## prune ##
            avg_similarity = (np.sum(similarity, 1) - 1) / (len(imp) - 1)
            prun_imp = imp / avg_similarity

            pr_criteria = np.sort(prun_imp)[-self.max_len]
            id_pr = np.where(prun_imp <= pr_criteria)[0]

            self.imp_pr = np.delete(imp, id_pr)
            pr_Doc_mat = np.delete(self.Doc_mat_, id_pr, 0)
            pr_input_document = np.delete(self.input_document, id_pr, 0)
        else:
            self.imp_pr = imp
            pr_Doc_mat = self.Doc_mat_
            pr_input_document = self.input_document

        similarity = self.pairwise_sent_similarity(pr_Doc_mat)
        similarity_list = self.return_similarity_list(similarity)

        sentences = ['sent_%d' % (i + 1) for i in range(len(self.imp_pr))]
        ys = ['sent_%dsent_%d' % (i + 1, j + 1) for i in range(len(self.imp_pr))
              for j in range(i + 1, len(self.imp_pr))]

        sentence_importance = dict(
            (s, self.imp_pr[i]) for i, s in enumerate(sentences))
        item_weights = dict((s, 1) for i, s in enumerate(sentences))
        sims = dict((s, similarity_list[i]) for i, s in enumerate(ys))

        x = LpVariable.dicts('sentence', sentences, 0, 1, LpBinary)
        y = LpVariable.dicts("y_(i,j)", ys, 0, 1, LpBinary)

        prob = LpProblem("knapsack", LpMaximize)

        # the objective
        cost = lpSum([sentence_importance[i] * x[i] for i in sentences])
        prob += cost
        cost2 = lpSum([-sims[i] * y[i] for i in ys])
        prob += cost2

        # constraints
        prob += lpSum([item_weights[i] * x[i] for i in sentences]) == self.N

        for i, s in enumerate(sentences[:-1]):
            for j, v in enumerate(sentences[i + 1:]):
                prob += lpSum(y[s + v] - x[s]) <= 0
                prob += lpSum(y[s + v] - x[v]) <= 0
                prob += lpSum(x[s] + x[v] - y[s + v]) <= 1

        prob.solve()
        self.obj = value(prob.objective)

        result = []
        for i in sentences:
            result.append(value(x[i]))

        self.result_y = []
        for i in ys:
            self.result_y.append(value(y[i]))

        self.selected_id = np.where(np.array(result) == 1)[0]
        output = np.take(pr_input_document, self.selected_id, 0)
        return output
