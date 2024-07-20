from training.supervised import Validator
from training.supervised.data_management import data_loader, DataProcessor
from training.supervised.models import FFNN, LSTM
from utils import loss_functions
import os, shutil, json, wandb, torch
from prodigyopt import Prodigy

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Learner(object):
    N_TICKERS = 10
    SEQ_LEN = 30
    TRADE_DURATION = 60
    BATCH_SIZE = 64
    LR = 3e-4
    LOG_WANDB = True

    def __init__(self, resume_from_checkpoint_folder=None, resume_wandb=True):
        resume_from_checkpoint_folder = os.path.join("E:","programming", "gw2_bot_data", "checkpoints","39000")
        # resume_wandb = False
        self.save_folder = os.path.join("E:","programming", "gw2_bot_data", "checkpoints")

        self.data_processor = DataProcessor(Learner.N_TICKERS, Learner.SEQ_LEN, Learner.TRADE_DURATION, Learner.BATCH_SIZE, n_processes=1)
        self.model = LSTM((DataProcessor.TICKER_LENGTH*Learner.N_TICKERS, Learner.SEQ_LEN), Learner.N_TICKERS)

        self.optim = Prodigy(self.model.parameters(), lr=1)
        # self.optim = torch.optim.Adam(self.model.parameters(), lr=Learner.LR)
        self.losses = []
        self.smooth_losses = []
        self.epoch = 0
        self.validator = Validator(seq_len=Learner.SEQ_LEN,
                                   ticker_length=DataProcessor.TICKER_LENGTH,
                                   n_tickers=Learner.N_TICKERS,
                                   model=self.model)

        self.wandb_run = None
        wandb_loaded = resume_from_checkpoint_folder is not None and self.load_checkpoint(resume_from_checkpoint_folder,
                                                                                          resume_wandb)

        if Learner.LOG_WANDB and self.wandb_run is None and not wandb_loaded:
            print("Attempting to create new wandb run")
            self.wandb_run = wandb.init(project="gw2-trading",
                                        group="quotient-prediction",
                                        reinit=True)
            print("Created new wandb run", self.wandb_run.id)

        # self.optim = torch.optim.Adam(self.model.parameters(), lr=Learner.LR)


    def learn(self):
        optim = self.optim
        dp = self.data_processor
        model = self.model
        smooth_loss = None

        for training_step in range(100000):
            x,y = dp.get_random_batch()

            optim.zero_grad()
            loss = loss_functions.quotient_prediction_loss(model, x, y, Learner.SEQ_LEN, Learner.TRADE_DURATION)
            # loss = loss_functions.quotient_prediction_loss(model, x, y, Learner.SEQ_LEN, Learner.TRADE_DURATION)
            # loss = loss_functions.tvd_loss(model, x, y)
            loss.backward()
            optim.step()

            l = loss.detach().cpu().item()
            if smooth_loss is None:
                smooth_loss = l
            else:
                smooth_loss = 0.99*smooth_loss + l*0.01

            if "cuda" in device:
                torch.cuda.empty_cache()
            epoch_log = {"Model Loss": smooth_loss}
            epoch_log["Learning Rate"] = self.adjust_lr()

            if self.epoch % 100 == 0:
                epoch_log["Simulated Profit"] = self.validator.test_model()

            if self.epoch % 100 == 0:
                self.save_checkpoint()

            if self.wandb_run is not None:
                self.wandb_run.log(epoch_log)
            self.epoch += 1
            epoch_log.clear()

    def adjust_lr(self):
        n = 0
        min_lr = 3e-5
        rate = 0.99995
        mean_lr = 0

        for group in self.optim.param_groups:
            if "lr" in group.keys():
                group["lr"] *= rate
                group["lr"] = max(group["lr"], min_lr)
                mean_lr += group["lr"]
                n += 1

        lr_report = mean_lr / n
        return lr_report

    def save_checkpoint(self):
        checkpoint_epoch = self.epoch
        folder_path = os.path.join(self.save_folder, str(checkpoint_epoch))
        os.makedirs(folder_path, exist_ok=True)

        print("Saving checkpoint {}...".format(checkpoint_epoch))
        existing_checkpoints = [int(arg) for arg in os.listdir(self.save_folder)]
        if len(existing_checkpoints) > 0:
            if max(existing_checkpoints) > checkpoint_epoch:
                print("found existing checkpoints in this folder, clearing folder before saving...")
                for checkpoint_name in existing_checkpoints:
                    shutil.rmtree(os.path.join(self.save_folder, str(checkpoint_name)))
            else:
                if len(existing_checkpoints) > 5:
                    existing_checkpoints.sort()
                    for checkpoint_name in existing_checkpoints[:-5]:
                        shutil.rmtree(os.path.join(self.save_folder, str(checkpoint_name)))

        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(folder_path, "MODEL.pt"))
        torch.save(self.optim.state_dict(), os.path.join(folder_path, "OPTIMIZER.pt"))

        book_keeping_vars = {"checkpoint_epoch": checkpoint_epoch}

        if self.wandb_run is not None:
            book_keeping_vars["wandb_run_id"] = self.wandb_run.id
            book_keeping_vars["wandb_project"] = self.wandb_run.project
            book_keeping_vars["wandb_entity"] = self.wandb_run.entity
            book_keeping_vars["wandb_group"] = self.wandb_run.group

        book_keeping_table_path = os.path.join(folder_path, "BOOK_KEEPING_VARS.json")
        with open(book_keeping_table_path, 'w') as f:
            json.dump(book_keeping_vars, f, indent=4)

        print("Checkpoint {} saved!\n".format(checkpoint_epoch))

    def load_checkpoint(self, folder_path, load_wandb):
        assert os.path.exists(folder_path), "UNABLE TO LOCATE FOLDER {}".format(folder_path)
        print("Loading from checkpoint at {}".format(folder_path))

        self.model.load_state_dict(torch.load(os.path.join(folder_path, "MODEL.pt")))
        self.optim.load_state_dict(torch.load(os.path.join(folder_path, "OPTIMIZER.pt")))

        wandb_loaded = False
        with open(os.path.join(folder_path, "BOOK_KEEPING_VARS.json"), 'r') as f:
            book_keeping_vars = dict(json.load(f))
            self.epoch = book_keeping_vars["checkpoint_epoch"]
            if "wandb_run_id" in book_keeping_vars.keys() and load_wandb:
                self.wandb_run = wandb.init(settings=wandb.Settings(start_method="spawn"),
                                            entity=book_keeping_vars["wandb_entity"],
                                            project=book_keeping_vars["wandb_project"],
                                            group=book_keeping_vars["wandb_group"],
                                            id=book_keeping_vars["wandb_run_id"],
                                            resume="allow",
                                            reinit=True)
                wandb_loaded = True

        print("Checkpoint loaded!")
        return wandb_loaded