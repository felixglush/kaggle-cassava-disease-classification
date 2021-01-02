from train_utilities.callbacks import CallbackGroupDelegator, Test1, Test2

cb = CallbackGroupDelegator([
            Test1(),
            Test2()
        ])

cb.fold_started(fold=3, metrics={})
cb.fold_ended(fold=3, metrics={"acc": 21})

cb.epoch_started(epoch=1, msg="hey test1", t=123)
cb.epoch_ended(epoch=1, msg="hey test1", t2=321)
