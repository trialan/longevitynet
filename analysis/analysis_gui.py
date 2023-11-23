import pygame
import torch
import numpy as np
import os

from longevitynet.modelling.model import ResNet50
from longevitynet.modelling.utils import (undo_min_max_scaling,
                                          unpack_model_input)


class AnalysisGUI:
    def __init__(self, dataset, dataloader, model):
        pygame.init()
        self.width, self.height = 600, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Life Expectancy Display with Pygame")
        self.dataloader = dataloader
        self.dataset = dataset
        self.index = 0
        self.model = model
        self.font = pygame.font.SysFont('Arial', 24)
        self.run()

    def run(self):
        running = True
        data_iter = iter(self.dataloader)
        item = next(data_iter)

        while running:
            for event in pygame.event.get():
                print(event)
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 or event.button == 3:
                        print("CLICK")
                        try:
                            item = next(data_iter)
                            self.index += 1
                        except StopIteration:
                            print("StopIteration")
                            self.index = 0
                            data_iter = iter(self.dataloader)
                            item = next(data_iter)

                self.screen.fill((0,0,0))

                # Assuming batch size of 1!
                img_path = self.dataset.image_paths[self.index]
                image = pygame.image.load(img_path)
                #image = pygame.transform.scale(image, (self.width, self.height))
                image_rect = image.get_rect()
                center_x = (self.width - image_rect.width) // 2
                center_y = (self.height - image_rect.height) // 2

                #self.screen.blit(image, (center_x, center_y))
                self.screen.blit(image, (center_x, 0))

                true_life_expectancy = item['life_expectancy']

                predicted_age = self.predict_life_expectancy(item)
                true_life_expectancy_text = self.font.render(f"True Life Expectancy: {true_life_expectancy.item():.2f} years", True, (255, 255, 255))
                predicted_age_text = self.font.render(f"Predicted Life Expectancy: {predicted_age:.2f} years", True, (255, 255, 255))
                self.screen.blit(true_life_expectancy_text, (10, self.height - 180))
                self.screen.blit(predicted_age_text, (10, self.height - 140))

                img_path = self.dataset.image_paths[self.index]
                person_name = os.path.basename(img_path).split('.')[0]
                person_name_text = self.font.render(f"Name: {person_name}", True, (255, 255, 255))
                self.screen.blit(person_name_text, (10, self.height - 100))

                difference = predicted_age - true_life_expectancy.item()
                difference_text = self.font.render(f"Difference: {difference:.2f} years", True, (255, 255, 255))
                self.screen.blit(difference_text, (10, self.height - 60))

                pygame.display.flip()
        pygame.quit()

    def predict_life_expectancy(self, item):
        model_input = unpack_model_input(item, "cpu")
        out = self.model(*model_input)
        out_in_years = convert_to_years(float(out), self.dataset)
        return out_in_years

def convert_to_years(raw_prediction, dataset):
    min_delta_value = np.min(dataset.deltas)
    max_delta_value = np.max(dataset.deltas)
    delta_prediction = undo_min_max_scaling(raw_prediction,
                                      min_val = min_delta_value,
                                      max_val = max_delta_value)
    prediction = delta_prediction + dataset.mean_life_expectancy
    return prediction

if __name__ == '__main__':
    import torch
    from longevitynet.modelling.config import CONFIG
    from longevitynet.modelling.data import get_dataset_dict

    CONFIG['BATCH_SIZE'] = 1 #necessary for iterating over test dataloader

    model_dir = "/Users/thomasrialan/Documents/code/longevitynet/deployment/"
    dataset_dict = get_dataset_dict(CONFIG)
    dataset = dataset_dict['datasets']['test']
    dataloader = dataset_dict['dataloaders']['test_dataloader']
    model = ResNet50()
    model.load_state_dict(torch.load(model_dir + "best_model_mae_6p3.pth"))
    app = AnalysisGUI(dataset, dataloader, model)



