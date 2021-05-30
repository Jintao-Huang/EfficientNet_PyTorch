import os


class Checker:
    def __init__(self, test_tester_dict, saver, check_epoch, ignore_num, logger=None):
        """

        :param test_tester_dict:
        :param saver:
        :param check_epoch: int. 每几个epoch check一次
        :param ignore_num: int. 前几次忽略的次数. 不check
        :param logger:
        """
        self.test_tester_dict = test_tester_dict
        self.saver = saver
        self.logger = logger
        self.check_epoch = check_epoch
        self.ignore_num = ignore_num  # ignore check_epoch
        self.best_test = 0.

    def step(self, epoch, last=False):
        if last or epoch % self.check_epoch == self.check_epoch - 1:
            if self.ignore_num > 0:
                self.ignore_num -= 1
                return
            best_test = []
            for k, test_tester in self.test_tester_dict.items():
                print("----------------------------- %s" % k)
                test_acc_dict = test_tester.test(last)
                if self.logger:
                    self.logger.log_mes({"%s_acc" % k: test_acc_dict})
                with open(os.path.join(self.saver.save_dir, "result.txt"), "a") as f:
                    f.write("%s_epoch%d_ACC: \n" % (k, epoch))
                    for label, acc in test_acc_dict.items():
                        f.write("    %s: %.4f%%\n" % (label, acc * 100))
                if k.lower() == "train":
                    continue
                test_total_acc = test_acc_dict['total_acc']
                best_test.append(test_total_acc)
            print("-----------------------------")
            best_test = sum(best_test) / len(best_test)
            if best_test >= self.best_test or last:
                self.best_test = best_test
                save_dir = self.saver.save_dir
                if last:
                    fname = "model_epoch%d_test%.4f_last.pth" % (epoch, best_test)
                else:  # not last
                    fname = "model_epoch%d_test%.4f.pth" % (epoch, best_test)
                    # 删除多余
                    for f in os.listdir(save_dir):
                        if f.endswith(".pth"):
                            path = os.path.join(save_dir, f)
                            print("Removed model %s..." % f, flush=True)
                            os.remove(path)
                self.saver.save(fname)
                print("Saved model %s..." % fname, flush=True)
