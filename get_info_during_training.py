# collect information at each epoch during training
# 1. loss
# 2. ndcg@50, recall@50
# 3. alignment and uniformity
# 4. preference scores of positive and negative items, given the user
# 5. NSPD/SPD of positive and negative items, given the user


import os
import math
import torch
from time import time
from recbole.trainer import Trainer
from recbole.utils import *
from tqdm import tqdm
from collections import defaultdict

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

        # collect information at each epoch during training
        # 1. validation loss
        # 2. ndcg@50, recall@50
        # 3. alignment and uniformity
        # 4. preference scores of positive and negative items, given the user
        # 5. NSPD/SPD of positive and negative items, given the user
        self.valid_loss_log = []
        self.valid_ndcg_log = []
        self.valid_recall_log = []
        self.valid_alignment_log = []
        self.valid_uniformity_log = []
        self.valid_pos_pref_log = defaultList(list)
        self.valid_pref_log = defaultList(list)
        self.valid_pos_nspd_log = defaultList(list)
        self.valid_nspd_log = defaultList(list)


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

    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        total_alignment = 0
        total_uniformity = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            user_e, item_e = self.model.forward(positive_u, positive_i)
            alignment = self.alignment(user_e, item_e)
            uniformity = self.uniformity(user_e) + self.uniformity(item_e)
            total_alignment += alignment
            total_uniformity += uniformity
            # scores: (batch_size, item_num) of all item scores for each user
            for user, score in zip(positive_u, scores):
                self.valid_pos_pref_log[user].append(score[positive_i])
                self.valid_pref_log[user].append(score) # actually all items
                # collect NSPD
                pos_nspd = self.model.data_dist_matrix[user, positive_i + self.tot_user_num]
                nspd = self.model.data_dist_matrix[user, self.tot_user_num:]
                self.valid_pos_nspd_log[user].append(pos_nspd)
                self.valid_nspd_log[user].append(nspd)
            


            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")

        self.valid_ndcg_log.append(result['ndcg@50'])
        self.valid_recall_log.append(result['recall@50'])

        # alignment and uniformity
        self.valid_alignment_log.append((total_alignment, total_alignment / num_sample))
        self.valid_uniformity_log.append((total_uniformity, total_uniformity / num_sample))

        return result
    
    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


     
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
                self.valid_loss_log.append(valid_result['loss'])

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result



import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
from logging import getLogger
from recbole.utils import init_logger, init_seed, get_model
from recbole.trainer import Trainer, NCLTrainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, prepare_DA_data_matrix
from pathlib import Path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=None, help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default=None, help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument("--DA_sampling", type=str, default=None, help="DA_sampling")
    parser.add_argument("--gpu_id", type=int, default=None, help="gpu id")
    parser.add_argument("--seed", type=int, default=2033, help="seed")
    parser.add_argument("--tau_da", type=float, default=1.0, help="tau_da")
    parser.add_argument("--beta_discount_factor", type=float, default=10, help="beta_discount_factor")

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    print(args.model, args.dataset, args.config_files, args.DA_sampling)

    config = Config(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    config['wandb_project'] = config['wandb_project'] + f'{args.seed}_{args.dataset}'
    
    print('wandb:', config['wandb_project'])
    print('backbone model:', config['model'], 'DA_sampling:', config['DA_sampling'])
    print('gpu_id:', config['gpu_id'], "device:", config['device'], torch.cuda.get_device_name(config['device']))
    print('tau for DA:', config['tau_da'])
    
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # prepare DA_data_matrix
    if config['DA_matrix_path'] is not None:
        train_data_matrix = np.load(config['DA_matrix_path'])['dist_array']
    else:
        try:
            train_data_matrix = np.load(os.path.join(config['data_path'], f'tau-{config["tau_da"]}_seed-{config["seed"]}_dist_data.npz'))['dists_array']
        except Exception as e:
            print(e)
            logger.info('DA_matrix_path is None, and DA_data_matrix is not found, so we will prepare DA_data_matrix')
            start = time.time()
            train_data_matrix = prepare_DA_data_matrix(train_data, config)
            logger.info(f'prepare DA_data_matrix cost {time.time() - start} s')
    start = time.time()
    train_data_matrix = torch.from_numpy(train_data_matrix).float()
    print('train_data_matrix:', train_data_matrix)
    logger.info(f'load DA_data_matrix cost {time.time() - start} s')

    # model loading and initialization
    print('model:', get_model(config["model"]))
    model = get_model(config["model"])(config, train_data._dataset, train_data_matrix).to(config['device'])
    print('device:', config['device'])
    logger.info(model)

    # trainer loading and initialization
    if config['model'] in ['NCL']:
        trainer = NCLTrainer(config, model) # NCL's trainer is special, haven't implement DA version for it
    else:
        trainer = DATrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=config["show_progress"])

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True)
    logger.info(f'best valid result: {best_valid_result}')
    logger.info(f'test result: {test_result}')
    best_valid_result_csv_format = pd.DataFrame(best_valid_result, index=[0]).to_csv(index=False, header=False)
    test_result_csv_format = pd.DataFrame(test_result, index=[0]).to_csv(index=False, header=False)

    # save result
