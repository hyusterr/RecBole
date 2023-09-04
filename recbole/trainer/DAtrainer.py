import math
from recbole.trainer import Trainer


# TODO: the beta from paper seems like always 1? need check
def calculate_beta_e(beta, beta_discount_factor, epoch_idx):
    if beta_discount_factor == 0:
        return 1
    else:
        beta_discount_factor = 1 / math.log(
                beta_discount_factor + epoch_idx, beta_discount_factor
        )
        return beta * beta_discount_factor


#TODO: 0. change parent class Trainer _train_epoch's return capture
#TODO: 1. add the code to train the model with the DA method 2. add the beta discount mechanism
#TODO: 3. deal with reselect_anchorset
class DATrainer(Trainer):

    def __init__(self, config, model):
        super(DATrainer, self).__init__(config, model)
        self.DA_sampling = config['DA_sampling']
        self.beta_discount_factor = config['beta_discount_factor']
        self.beta = config['beta']
        
        # overwrite the saved_model_file
        saved_model_file = "{}-{}-{}.pth".format(self.config["model"], self.config["DA-sampling"], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)



    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):

        main_loss_func = self.model.calculate_loss
        if self.DA_sampling == 'full':
            da_loss_func = self.model.calculate_loss_DA_full
        elif self.DA_sampling == 'ui':
            da_loss_func = self.model.calculate_loss_DA_ui
        elif self.DA_sampling == 'anchor':
            da_loss_func = self.model.calculate_loss_DA_anchor
        elif self.DA_sampling == 'none':
            da_loss_func = None
        
        self.model.train()
        total_loss = 0.
        main_loss = 0.
        da_loss = 0.
        
        beta_e = calculate_beta_e(self.beta, self.beta_discount_factor, epoch_idx)

        for batch_idx, interaction in enumerate(train_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            main_loss_value = main_loss_func(interaction)
            if da_loss is not None:
                da_loss_value = da_loss_func(interaction)
            else:
                da_loss_value = 0

            loss = main_loss_value + beta_e * da_loss_value
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            main_loss += main_loss_value.item()
            da_loss += da_loss_value.item()

        return total_loss, main_loss, da_loss, beta_e
