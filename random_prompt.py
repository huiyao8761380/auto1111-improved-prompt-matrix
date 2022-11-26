import re
import random
import gradio as gr
import modules.shared as shared
import modules.scripts as scripts
import modules.sd_samplers
from modules.processing import process_images, StableDiffusionProcessingTxt2Img


class Script(scripts.Script):
    def title(self):
        return "random prompt 0.1"

    def ui(self, is_img2img):
        dummy = gr.Textbox(label="random prompt script has been started",value="tag example:,<short hair|long hair|messy hair>;   Batch count>=2,Batch size=1")
        #sametag = gr.Checkbox(label="Same tag can be generated.", value=False)
        #norand = gr.Checkbox(label="Not random,do each prompt", value=False)
        return [dummy]

    def run(self, p, dummy):
        #modules.processing.fix_seed(p)

        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

        all_prompts = [original_prompt]
        #p_prompt=all_prompts[0]
        #之前只能叠4层到10w

        #p.prompt[0].
        ####datafinish=False
        for this_prompt in all_prompts:
            for data in re.finditer(r'(<([^>]+)>)', this_prompt):
                if data:
                    ####datafinish=True
                    span = data.span(1)
                    new_prompt = this_prompt[:span[0]]
                    gen_prompt = this_prompt[span[0]:]
                    #print(f"new_prompt：{new_prompt}")
                    #print(f"gen_prompt：{gen_prompt}")
                    break
        
        print(f"new_prompt：{new_prompt}")
        print(f"gen_prompt：{gen_prompt}")
        
        ####if datafinish==True:
        #for i_prompt in all_prompts:
        for ip in range(p.n_iter+1):
            
            my_prompt=""
            #all_prompts.remove(i_prompt)
            for data in re.finditer(r'(<([^>]+)>)', gen_prompt):
                if data:
                    
                    items = data.group(2).split("|")#=list
                    length=len(items)-1
                    rand_item = items[random.randint(0,length)]
                    my_prompt=my_prompt+rand_item.strip()+","
            i_prompt=new_prompt+my_prompt
            all_prompts.append(i_prompt)
            print(f"i_prompt：{i_prompt}")
            print(f"all_prompts len：{len(all_prompts)}")
 
        all_prompts.remove(all_prompts[0])


        
        

        p.prompt = all_prompts * p.n_iter #all_prompts * p.n_iter
        #else:
        #p.prompt = out_prompts*p.n_iter
        #p.n_iter = total_images
        #print(f"p.n_iter：{p.n_iter}") #=batch_count
        p.do_not_save_grid = True
        p.prompt_for_display = original_prompt

        return process_images(p)
