from .loss_utils.l12 import l1, l2, complex_l2, complex_l1

tag = 'loss_full'


def default_loss(S0_pred, S0, k_pred, k, theta_pred, theta, **kwargs):
    S0_l1_lambda = kwargs.get('S0_l1_lambda', 1)
    S0_l1_loss = l1(S0_pred, S0) * S0_l1_lambda
    print(f'in {tag}, S0_l1_loss: {S0_l1_loss.item()}')

    S0_l2_lambda = kwargs.get('S0_l2_lambda', 1)
    S0_l2_loss = l2(S0_pred, S0) * S0_l2_lambda
    print(f'in {tag}, S0_l2_loss: {S0_l2_loss.item()}')

    k_complex_l1_lambda = kwargs.get('k_complex_l1_lambda', 1)
    k_complex_l1_loss = complex_l1(k_pred, k) * k_complex_l1_lambda
    print(f'in {tag}, k_complex_l1_loss: {k_complex_l1_loss.item()}')

    k_complex_l2_lambda = kwargs.get('k_complex_l2_lambda', 1)
    k_complex_l2_loss = complex_l2(k_pred, k) * k_complex_l2_lambda
    print(f'in {tag}, k_complex_l2_loss: {k_complex_l2_loss.item()}')

    theta_l1_lambda = kwargs.get('theta_l1_lambda', 1)
    theta_l1_loss = l1(theta_pred, theta) * theta_l1_lambda
    print(f'in {tag}, theta_l1_loss: {theta_l1_loss.item()}')

    theta_l2_lambda = kwargs.get('theta_l2_lambda', 1)
    theta_l2_loss = l2(theta_pred, theta) * theta_l2_lambda
    print(f'in {tag}, theta_l2_loss: {theta_l2_loss.item()}')

    return S0_l1_loss + S0_l2_loss + k_complex_l1_loss + k_complex_l2_loss + theta_l1_loss + theta_l2_loss
