#Standard variables
init="haar"
m = 10
detector_positions = (33, 66)
values=np.array([1,7])

#Sweep Array:
sweep=[]                        #<----------------------------------------
sweep_label=""                  #<----------------------------------------

#Perform training
trainers = []
for s in sweep:
    cfg = TrainConfig(m_side=m, theta_enc=1, normalize_energy=True, encoding='amplitude', 
            detectors=detector_positions, loss_kind='mse', learning_rate=0.1, batch_size=64,
                      init=init, max_epochs=35, patience=2, min_delta=1e-4,     #patience=2 und min_delta=1e-4 hat sich für mich jetzt gut ergeben, vielleicht sogar nur 1e-3
                      param_init_seed=1550, eta_bs=1.0, alpha_fiber=0.0)
    E_tr, y_tr, E_te, y_te = get_data(values, number=2000, m_side=m,
                                      theta_enc=1, normalize_energy=True,
                                      seed=1550, balanced=True, verbose=True)
    t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
    t.fit(E_tr, y_tr)
    t.test_acc = t.evaluate(E_te, y_te)[1]  
    t.save(f"results/{sweep_label}/{s}.json", test_acc=t.test_acc)
    trainers.append(t)

#Plot trainingresults
fig, _ = plot_training(trainers, sweep, keys=("batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label)

for s, t in zip(sweep, trainers):
    print(f"{sweep_label}={s}  epochs={len(t.history['loss']):3d}"
          f"loss={t.history['loss'][-1]:.5f}  "
          f"train acc={t.history['acc'][-1]:.4f}  "
          f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

#%%%
plt.savefig(f"results/{sweep_label}/plot_training.png", dpi=600, bbox_inches='tight')