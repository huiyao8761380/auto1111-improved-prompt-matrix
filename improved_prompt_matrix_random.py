import re
import random
import gradio as gr
import modules.shared as shared
import modules.scripts as scripts
import modules.sd_samplers
from modules.processing import process_images, StableDiffusionProcessingTxt2Img


class Script(scripts.Script):
    def title(self):
        return "Improved prompt matrix random"

    def ui(self, is_img2img):
        dummy = gr.Checkbox(label="Usage: <corgi|cat>,<goggles|a hat>")
        return [dummy]

    def run(self, p, dummy):
        #modules.processing.fix_seed(p)

        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

        matrix_count = 0
        prompt_matrix_parts = []
        for data in re.finditer(r'(<([^>]+)>)', original_prompt):
            if data:
                matrix_count += 1
                span = data.span(1)
                items = data.group(2).split("|")
                prompt_matrix_parts.extend(items)

        all_prompts = [original_prompt]
        
        while True:
            found_matrix = False
            for this_prompt in all_prompts:
                for data in re.finditer(r'(<([^>]+)>)', this_prompt):
                    if data:
                        found_matrix = True
                        # Remove last prompt as it has a found_matrix
                        all_prompts.remove(this_prompt)
                        span = data.span(1)
                        items = data.group(2).split("|")
                        #####length=len(items)-1
                        #####for item in items:
                            #####rand_item = items[random.randint(0,length)]
                            #####new_prompt = this_prompt[:span[0]] + rand_item.strip() + this_prompt[span[1]:]#this_prompt=全长的prompt
                            #####all_prompts.append(new_prompt.strip())
                            ###my_prompts.append(new_prompt.strip())
                            #####print(f"new prompt：{new_prompt}")
                            #print(f"p.n_iter：{p.n_iter}")#=count
                        for item in items:
                            new_prompt = this_prompt[:span[0]] + item.strip() + this_prompt[span[1]:]#通过strip()去掉首尾空白字符 .span返回查找字的(首,尾)
                            all_prompts.append(new_prompt.strip())
                            ######print(f"new prompt：{new_prompt}")
                    break
                if found_matrix:
                    break
            if not found_matrix:
                ###promptlength=len(my_prompts)-1
                ###rand_prompt = my_prompts[random.randint(0,promptlength)]
                ###all_prompts.append(rand_prompt)
                break

        promptlength=len(all_prompts)-1
        out_prompts = []
        #rand_prompt = all_prompts[random.randint(0,promptlength)]
        for my_prompt in all_prompts: 
            rand_i=random.randint(0,promptlength)
            out_prompts.append(all_prompts[rand_i])
            all_prompts.remove(all_prompts[rand_i])
            promptlength=len(all_prompts)-1
            
        #print(f"out_prompts：{out_prompts}")

        total_images = len(all_prompts) * p.n_iter
        print(f"Prompt matrix will create {total_images} images")

        total_steps = p.steps * total_images
        if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
            total_steps *= 2
        shared.total_tqdm.updateTotal(total_steps)

        
        
        
        #all_prompts.append(rand_prompt)

        p.prompt = out_prompts*p.n_iter #all_prompts * p.n_iter
        #p.seed =-1#[item for item in range(int(p.seed), int(p.seed) + p.n_iter) for _ in range(len(all_prompts))]
        p.n_iter = total_images
        #print(f"p.n_iter：{p.n_iter}") #=batch_count
        p.do_not_save_grid = True
        p.prompt_for_display = original_prompt

        return process_images(p)
