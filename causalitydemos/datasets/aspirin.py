import numpy as np


class LognormalAspirinModel:
    def __init__(self, a, b, c, severity_mean=0., severity_sigma=1.0, dosage_sigma=1.0,
                 duration_sigma=1.0):
        """An aspirin model that draws examples from a lognormal distribution
        (see thesis).

        Args:
            a (float): Parameter that controls the dosage's dependence on headache severity
            b (float): Parameter that controls headache duration's dependence on headache severity
            c (float): Parameter that controls headache duration's dependence on dosage of aspirin
        """
        self.a = a
        self.b = b
        self.c = c
        self.severity_mean = severity_mean
        self.severity_sigma = severity_sigma
        self.dosage_sigma = dosage_sigma
        self.duration_sigma = duration_sigma
 
    def generate_examples(self, num_examples=1):
        z = self._sample_z(size=[num_examples])
        t = self._sample_t_cond_on_z(z=z)
        y = self._sample_y_cond_on_zt(z=z, t=t)
        return z, t, y

    def _sample_z(self, size=None):
        """Sample headache severity
        """
        return np.random.lognormal(mean=self.severity_mean, sigma=self.severity_sigma, size=size)

    def _sample_t_cond_on_z(self, z):
        """Sample asprin dosage
        """
        size = None if isinstance(z, float) else z.shape
        return (z**self.a) * np.random.lognormal(mean=0.0, sigma=self.dosage_sigma, size=size)

    def _sample_y_cond_on_zt(self, z, t):
        """Sample headache severity
        """
        size = None if isinstance(z, float) else z.shape
        return (z**self.b / t**self.c) * np.random.lognormal(mean=0.0, sigma=self.duration_sigma,
                                                             size=size)

    def sample_interventional(self, dose, num_examples=1):
        assert dose > 0
        z = self._sample_z(size=[num_examples])
        t = np.ones([num_examples]) * dose
        y = self._sample_y_cond_on_zt(z=z, t=t)
        return z, t, y 

    def calc_exp_y_cf_given_ty(self, t, y, t_cf):
        return calc_exp_y_cf_given_ty(
            t, y, t_cf, self.a, self.b, self.c, self.severity_mean,
            self.severity_sigma, self.dosage_sigma, self.duration_sigma)


def calc_exp_y_cf_given_ty(t, y, t_cf, a, b, c, severity_mean, severity_sigma, dosage_sigma, duration_sigma):
    d = b - a * c
    # The covariance matrix for the variables T, Y
    cond_cov_submatrix = np.array(
        [[a**2 * severity_sigma**2 + dosage_sigma**2, a*d*severity_sigma**2-c*dosage_sigma**2],
         [a*d*severity_sigma**2-c*dosage_sigma**2, d**2*severity_sigma**2 + c**2*dosage_sigma**2 + duration_sigma**2]])
    # cross-covariance term between Y_cf and T, Y
    cross_cov_submatrix = np.array(
        [[a*b*severity_sigma**2], [b*d*severity_sigma**2]]
    )
    mu_y_cf = (b*severity_mean - c*np.log(t_cf)) 
    mu_ty = np.array([a, d])*severity_mean
    cond_cov_submatrix_inv = np.linalg.inv(cond_cov_submatrix)
    # Calculate the log-normal gaussian mu parameter (not same as mean)
    mu_y_cf_given_ty = mu_y_cf + np.matmul(
        cross_cov_submatrix.T, np.matmul(cond_cov_submatrix_inv, (np.log(np.array([t, y])) - mu_ty)))
    # Calculate the covariance
    sigma_squared_y_cf_given_ty = (b**2*severity_sigma**2 + duration_sigma**2) - np.matmul(
        cross_cov_submatrix.T, np.matmul(cond_cov_submatrix_inv, cross_cov_submatrix))
    # Calculate the mean of the lognormal variable
    return np.exp(mu_y_cf_given_ty + sigma_squared_y_cf_given_ty / 2)
    
    
    