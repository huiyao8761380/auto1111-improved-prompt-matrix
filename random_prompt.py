import re
import random
import gradio as gr
import modules.shared as shared
import modules.scripts as scripts
import modules.sd_samplers
from modules.processing import process_images, StableDiffusionProcessingTxt2Img

#https://github.com/huiyao8761380/random-prompt
# ,1girl,【short|long|messy】 hair,【smile|blush|sad】,
#Random prompt v0.7 因需要使用LORA模型的原因文本替换了'<|||>'为'【|||】'
#能力有限-务必将随机tag置于末尾,因为就是这么设计的...

class Script(scripts.Script):
    def title(self):
        return "Random prompt"

    def ui(self, is_img2img):
        dummy = gr.Textbox(label="random prompt script has been started",value="Batch count >=1,Batch size=1,random tags are recommended at the end:  ,【short|long|messy】 hair, ")
        sameseed = gr.Checkbox(label="Same seed generated.", value=False)
        return [dummy,sameseed]

    def run(self, p, dummy, sameseed):

        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

        all_prompts = [original_prompt]
        

        split_str=['']
        right_str=[]
        gen_prompt=''
        new_prompt=''
        
        for this_prompt in all_prompts:
            for data in re.finditer(r'(【([^】]+)】)', this_prompt):
                if data:
                    span = data.span(1)
                    new_prompt = this_prompt[:span[0]]
                    gen_prompt = this_prompt[span[0]:]
                    this_str=gen_prompt.replace("【","】")
                    split_str =this_str.split("】")
                    break
        
        #print(f"new_prompt：{new_prompt}")
        #print(f"gen_prompt：{gen_prompt}")
        
        
        split_str[0]=split_str[0].strip()
        split_str[0]=split_str[0].strip(',')
        split_str[0]=split_str[0]+' '
        for in_str in split_str:
            if "|" not in in_str:
                if "," in in_str:
                    FL_str=in_str.split(",")
                    right_str.append(FL_str[0])
                    right_str.append(FL_str[1])
                else:
                    right_str.append(in_str)
        #print (right_str)
        #print (len(right_str))
        #d=int((len(right_str)-1)/2)
        #print(d)
        
        for ip in range(p.n_iter+1):
            
            my_prompt=""
            
            i=0
            for data in re.finditer(r'(【([^】]+)】)', gen_prompt):
                if data:

                    items = data.group(2).split("|")#=list
                    length=len(items)-1

                    rand_item = items[random.randint(0,length)]
                    
                    rand_str=right_str[i]+rand_item+right_str[i+1]
                    
                    my_prompt=my_prompt+rand_str.strip()+","
                    i=i+2
                    
            i_prompt=new_prompt+my_prompt
            all_prompts.append(i_prompt)
            #print(f"i_prompt：{i_prompt}")
            #print(f"all_prompts len：{len(all_prompts)}")
            
        if gen_prompt != '':
            all_prompts.remove(all_prompts[0])
        

        p.prompt = all_prompts * p.n_iter #all_prompts * p.n_iter
        if sameseed == True:
            p.seed =[item for item in range(int(p.seed), int(p.seed) + p.n_iter) for _ in range(len(all_prompts))]
        #print(f"p.n_iter：{p.n_iter}") #=batch_count
        p.do_not_save_grid = True
        p.prompt_for_display = original_prompt

        return process_images(p)
