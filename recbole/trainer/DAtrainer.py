import os
import math
import torch
from time import time
from recbole.trainer import Trainer
from recbole.utils import *
from tqdm import tqdm

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
        
        # special design for SGL and Direct, since the loss scale varies too big
        if config['model'] == 'SGL': # main loss is at the scale of 1e5
            config['beta'] *= 1000
        if config['model'] == 'DirectAU': # main loss is at the scale of 1e4
            config['beta'] *= 10

        super(DATrainer, self).__init__(config, model)
        self.DA_sampling = config['DA_sampling']
        self.beta_discount_factor = config['beta_discount_factor']
        self.beta = config['beta']
        
        # overwrite the saved_model_file
        saved_model_file = "{}-{}-{}-{}.pth".format(self.config['dataset'], self.config["model"], self.config["DA_sampling"], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        # if is using NCL
        self.is_NCL = False
        if config['model'] == 'NCL':
            self.num_m_step = config["m_step"]
            self.is_NCL = True
            assert self.num_m_step is not None



    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):

        self.model.train()
 
        main_loss_func = self.model.calculate_loss
        if self.DA_sampling == 'full':
            da_loss_func = self.model.calculate_loss_DA_full
        elif self.DA_sampling == 'ui':
            da_loss_func = self.model.calculate_loss_DA_ui
        elif self.DA_sampling == 'anchor':
            da_loss_func = self.model.calculate_loss_DA_anchor
        elif self.DA_sampling == 'none':
            da_loss_func = None
        
        total_loss = 0.
        main_loss = 0.
        da_loss = 0.
        
        beta_e = calculate_beta_e(self.beta, self.beta_discount_factor, epoch_idx)
        
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            main_loss_value, reg_loss_with_weight = main_loss_func(interaction)

            if isinstance(main_loss_value, tuple):
                main_loss_value = sum(main_loss_value)
            
            # need to reselect anchorset for the first batch when using anchor sampling
            if self.DA_sampling == 'anchor' and batch_idx == 0:
                da_loss_value = da_loss_func(interaction, reselect_anchor=True)  
            elif da_loss_func is not None:
                da_loss_value = da_loss_func(interaction)
            else:
                da_loss_value = torch.tensor(0.0)

            loss = beta_e * main_loss_value + (1 - beta_e) * da_loss_value + reg_loss_with_weight
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            main_loss += main_loss_value.item()
            da_loss += da_loss_value.item()

        return total_loss, main_loss, da_loss, beta_e

    
    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):

            # if is using NCL
            if self.is_NCL and epoch_idx % self.num_m_step == 0:
                self.logger.info("Running E-step ! ")
                self.model.e_step()

            # train
            training_start_time = time()
            # modify here to capture main_loss, da_loss, beta_e
            train_loss, main_loss, da_loss, beta_e = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            # modify here to add main_loss, da_loss, beta_e, the function can eat tuple
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, (train_loss, main_loss, da_loss, beta_e)
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, (train_loss, main_loss, da_loss, beta_e))
            self.wandblogger.log_metrics(
                {
                    "epoch": epoch_idx, 
                    "train_loss": train_loss, 
                    "train_step": epoch_idx, 
                    "main_loss": main_loss, 
                    "da_loss": da_loss, 
                    "beta_e": beta_e
                },
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result
