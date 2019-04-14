import configs
import argparse
from runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='Training/testing for Face Generation.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='\'train\' or \'test\' mode.')
    return parser.parse_args()

def create_dir():
    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

if __name__ == "__main__":
    # Parse args.
    args = parse_args()
    create_dir()
    runner = Runner()
    n_epochs = configs.n_epochs
    if args.mode == 'train':
        for epoch in range(n_epochs):
            print('Epoch: %d/%d' % (epoch+1,n_epochs))
            d_loss, g_loss = runner.train_model()
            # Checkpoint the model after each epoch.
            d_loss, g_loss= '%.3f'%(d_loss), '%.3f'%(g_loss)
            model_path = os.path.join(configs.model_save_dir, \
                        'model_{}_d_{}_g_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), d_loss, g_loss))
            torch.save(model.state_dict(), model_path)
            print('='*20)
            scheduler.step(val_loss)
    else:
        runner.test_model()
        print('='*20)
