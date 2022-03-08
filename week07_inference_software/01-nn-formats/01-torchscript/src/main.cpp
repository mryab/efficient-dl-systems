#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("./vgg16.pt");

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::ones({1, 3, 224, 224}));

        at::Tensor output = module.forward(inputs).toTensor();

        float* itr = (float*)output.data_ptr();
        std::cout << output.sizes() << std::endl;

        int argmax = 0;
        float value = 0;
        for (int i =0; i < output.sizes()[0] * output.sizes()[1]; i++) {
            if(*itr > value) {
                value = *itr;
                argmax = i;
            }
            itr++;
        }

        std::cout << "Answer = " << argmax << " with logit = " << value << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";
}
