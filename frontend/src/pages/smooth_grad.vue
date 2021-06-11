<template>
    <div class="page" v-loading="loading" element-loading-background="rgba(0, 0, 0, 0.45)">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class="value" size="small" v-model="params.model" @change="update">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
                </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="params.target = images[params.input]">
                    <el-option v-for="image in Object.keys(images)" :key="images[image]" :value='image'/>
                </el-select>
            </div>
            <div class="item"><div class="title">target</div><el-input class="value" size="small" type='number' v-model="params.target"  @change="update"/></div>
            <div class="item"><div class="title">epochs</div><el-input class="value" size="small" type='number' v-model="params.epochs"  @change="update"/></div>
            <div class="item"><div class="title">noise level</div><el-input class="value" size="small" type='number' v-model="params.noise_level"  @change="update"> <template slot="append">%</template></el-input></div>
            <div class="item"><div class="title"></div><el-button icon='el-icon-refresh' type="primary" size="small" circle  @click="update"/></div>
        </div>
        <div class="network">
            <div class="input">
                <div class="image-wrapper">
                    <el-image :src="params.input">
                        <div slot="error" class="image-slot">
                            <i class="el-icon-lollipop"></i>
                        </div>
                    </el-image>
                    <div class="caption">Input</div>
                </div>
            </div>
            <div class="sliency">
                <div class="image-wrapper">
                    <el-image :src="res.colorful">
                        <div slot="error" class="image-slot">
                            <i class="el-icon-lollipop"></i>
                        </div>
                    </el-image>
                    <div class="caption">Smooth Gradient</div>
                </div>
                <div class="image-wrapper">
                    <el-image :src="res.grayscale">
                        <div slot="error" class="image-slot">
                            <i class="el-icon-lollipop"></i>
                        </div>
                    </el-image>
                    <div class="caption">Smooth Saliency</div>
                </div>
                <div class="image-wrapper">
                    <el-image :src="res.grad_x_image">
                        <div slot="error" class="image-slot">
                            <i class="el-icon-lollipop"></i>
                        </div>
                    </el-image>
                    <div class="caption">Smooth Saliency * Image</div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'saliency',
    data() {
        return {
            models: [],
            images: {},
            activations: [],
            params: {
                model: 'vgg19',
                input: '',
                target: null,
                epochs: 50,
                noise_level: 10
            },
            res: {},
            loading: false
        };
    },
    created() {
        this.config()
    },
    sockets: {
        models(data) {
            this.models = data
        },
        images(data) {
            this.images = data

            this.params.input = Object.keys(data)[0]
            this.params.target = this.images[this.params.input]
        },
        response_smooth_grad(data) {
            this.res = data
            this.loading = false
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
            this.$socket.emit('get_images')
        },
        update() {
            this.$socket.emit("smooth_grad", this.params);
            this.loading = true
        }
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        display: flex;
        flex-flow: row;

        align-items: center;
        justify-items: center;

        .input {
            flex: 0 0 auto;
        }

        .sliency {
            flex: 1 1 auto;
            display: flex;
            flex-flow: column;
            align-items: center;
            justify-items: center;
            img {
                padding: 10px;
            }
        }
    }
}

</style>
