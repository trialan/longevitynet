import pygame
import torch
import numpy as np
import os

from longevitynet.modelling.model import ResNet50
from longevitynet.modelling.data import generate_dataset
from longevitynet.modelling.utils import undo_min_max_scaling


class AnalysisGUI:
    def __init__(self, dataset, model):
        pygame.init()
        self.width, self.height = 600, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Life Expectancy Display with Pygame")
        self.dataset = dataset
        self.index = 0
        self.model = model
        self.font = pygame.font.SysFont('Arial', 24)
        self.run()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        self.index += 1
                        if self.index >= len(self.dataset):
                            self.index = 0
                    elif event.key == pygame.K_LEFT:
                        self.index -= 1
                        if self.index < 0:
                            self.index = len(self.dataset) - 1
            self.screen.fill((0,0,0))

            image_tensor, _, _, _ = self.dataset[self.index]
            image_np = image_tensor.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image = pygame.surfarray.make_surface(image_np)
            image = pygame.transform.rotate(image, 90*3)



            self.screen.blit(image, (0, 0))
            _, _, true_life_expectancy, _ = self.dataset[self.index]
            predicted_age = self.predict_life_expectancy()
            true_life_expectancy_text = self.font.render(f"True Life Expectancy: {true_life_expectancy.item():.2f} years", True, (255, 255, 255))
            predicted_age_text = self.font.render(f"Predicted Life Expectancy: {predicted_age:.2f} years", True, (255, 255, 255))
            self.screen.blit(true_life_expectancy_text, (10, self.height - 180))
            self.screen.blit(predicted_age_text, (10, self.height - 140))

            # Extract name from the image path
            img_path = self.dataset.image_paths[self.index]
            person_name = os.path.basename(img_path).split('.')[0]
            person_name_text = self.font.render(f"Name: {person_name}", True, (255, 255, 255))
            self.screen.blit(person_name_text, (10, self.height - 100))

            difference = predicted_age - true_life_expectancy.item()
            difference_text = self.font.render(f"Difference: {difference:.2f} years", True, (255, 255, 255))
            self.screen.blit(difference_text, (10, self.height - 60))

            pygame.display.flip()
        pygame.quit()

    def predict_life_expectancy(self):
        image, _, _, _ = self.dataset[self.index]
        out = self.model(image.unsqueeze(0))
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
    from life_expectancy.modelling.train import DS_VERSION
    import torch
    model_dir = "/Users/thomasrialan/Documents/code/longevity_project/saved_model_binaries/"
    dataset = generate_dataset(DS_VERSION)
    model = ResNet50()
    model.load_state_dict(torch.load(model_dir + "best_model_20231013-145803_0p01107617188245058.pth"))
    app = AnalysisGUI(dataset, model)
